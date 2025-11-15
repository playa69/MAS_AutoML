"""Агент для поиска и извлечения информации о AutoML фреймворках из научных статей."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, TypedDict
from urllib.error import URLError

import arxiv
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from openai import AuthenticationError
from pypdf import PdfReader

from ...config.settings import LLMConfig, settings
from .models import AutoMLRelevance, FrameworkInfo, FrameworksList, PaperInfo, VerificationFeedback

logger = logging.getLogger(__name__)

# Попытка импортировать Tavily (опционально)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None  # type: ignore


class PaperSearchState(TypedDict, total=False):
    """Состояние для графа поиска и извлечения информации о фреймворках."""

    query: str
    papers: list[dict[str, Any]]
    paper_frameworks: list[dict[str, Any]]
    error: str | None
    sort_by: arxiv.SortCriterion | None
    days_back: int | None
    max_results: int | None
    frameworks: list[dict[str, Any]]
    filtered_papers: list[dict[str, Any]]  # Отфильтрованные статьи после анализа
    extraction_iteration: int  # Счетчик итераций извлечения
    verification_feedback: str | None  # Текстовый фидбек от верификатора для улучшения извлечения
    verification_score: int | None  # Оценка качества последнего извлечения (0-100)
    verification_need_rerun: bool  # Временно для условной функции (не сохраняется между нодами)


class PaperSearchTools:
    """Инструменты для поиска научных статей."""

    @staticmethod
    async def search_arxiv(
        query: str,
        max_results: int = 5,
        sort_by: arxiv.SortCriterion | None = None,
        days_back: int | None = None,
    ) -> list[PaperInfo]:
        """Поиск статей в arXiv."""
        try:
            sort_by = sort_by or arxiv.SortCriterion.Relevance
            cutoff_date = (datetime.now() - timedelta(days=days_back)) if days_back else None
            search_max_results = max_results * 3 if days_back else max_results
            
            def _search():
                return list(arxiv.Search(query=query, max_results=search_max_results, sort_by=sort_by).results())

            papers = await asyncio.get_event_loop().run_in_executor(None, _search)
            results = []

            for paper in papers:
                if cutoff_date and paper.published:
                    paper_date = paper.published.replace(tzinfo=None) if paper.published.tzinfo else paper.published
                    if paper_date < cutoff_date:
                        continue
                elif cutoff_date:
                    continue
                
                results.append(
                    PaperInfo(
                        title=paper.title,
                        authors=[author.name for author in paper.authors],
                        abstract=paper.summary,
                        url=paper.entry_id,
                        source="arxiv",
                        pdf_url=paper.pdf_url,
                        published_date=str(paper.published) if paper.published else None,
                    )
                )
                
                if len(results) >= max_results:
                    break
            
            return results
        except (URLError, ValueError, OSError, Exception):
            return []


class PaperAnalyzer:
    """Класс для анализа и фильтрации научных статей."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self.llm = llm

    async def analyze_paper_relevance(self, paper_info: PaperInfo) -> tuple[bool, str]:
        """Анализирует релевантность статьи запросу.
            
        Returns:
            (is_relevant, reason) - релевантна ли статья и причина (если не релевантна)
        """
        if not self.llm:
            # Без LLM считаем все статьи релевантными
            return True, ""

        system_prompt = """
        Ты ассистент-классификатор научных статей.
        Твоя задача — определить, описывается ли в статье AutoML-фреймворк/система/библиотека/платформа
        для автоматизации задач машинного обучения (подбор моделей, гиперпараметров, пайплайнов, NAS и т.п.).

        Статья считается РЕЛЕВАНТНОЙ (label = "ДА") ТОЛЬКО если выполняется хотя бы одно:
        1) В статье представляют или описывают AutoML-фреймворк, библиотеку, платформу или систему
        (например, Auto-sklearn, H2O AutoML, AutoKeras, TPOT, AutoGluon и другие аналогичные решения).
        2) В статье описан метод/подход, встроенный в AutoML-пайплайн
        (автоматический выбор моделей, автоматический выбор признаков, автоматизированный дизайн пайплайнов
        или нейроархитектур (NAS), гиперпараметрический поиск и т.п.), и это рассматривается как часть
        AutoML-системы или фреймворка.
        3) LLM используется как часть AutoML-системы (например, LLM управляет поиском моделей, генерацией
        пайплайнов, конфигураций, автоматизацией экспериментов и т.п.), а не просто как отдельная модель.

        Статья считается НЕ РЕЛЕВАНТНОЙ (label = "НЕТ"), если:
        - Она про большие языковые модели (LLM) как отдельные модели,
        например GPT-4o, GPT-4o-mini, Llama-3.3-70B, Meta-Llama-3.1-8B, Mistral-Nemo-2407, Phi-4,
        Qwen2.5-32B и другие, без явного описания AutoML-фреймворка/системы.
        - Она только про обучение, дообучение, тонкую настройку, сжатие или оценку LLM-моделей,
        без автоматизации выбора моделей/гиперпараметров/пайплайнов.
        - Она в целом про машинное обучение, нейронные сети, оптимизацию, NLP и т.п., но не описывает
        AutoML-фреймворк, систему или подход в контексте AutoML.

        Если статья НЕ РЕЛЕВАНТНА:
        - label = "НЕТ"
        - reason — кратко (1–2 предложения), почему статья не подходит.

        Если статья РЕЛЕВАНТНА:
        - label = "ДА"
        - reason можно опустить или оставить пустой.

        Всегда отвечай только в формате JSON, строго согласно заданной схеме.
        """

        human_prompt = f"""
        Определи, релевантна ли эта научная статья запросу об AutoML-фреймворках.

        Название статьи: {paper_info.title}
        Авторы: {', '.join(paper_info.authors)}
        Абстракт (фрагмент, до 1000 символов):
        {paper_info.abstract[:1000]}
        """

        try:
            structured_llm = self.llm.with_structured_output(AutoMLRelevance)
            result: AutoMLRelevance = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )

            is_relevant = result.label == "ДА"
            reason = result.reason or ""

            if not is_relevant:
                logger.info(f"Статья отфильтрована: {reason}")

            return is_relevant, reason
        except Exception as e:
            logger.warning(f"Ошибка при анализе статьи '{paper_info.title}': {e}")
            # В случае ошибки считаем статью релевантной, чтобы не потерять данные
            return True, ""


class FrameworkVerifier:
    """Класс для верификации качества извлеченных данных о фреймворках."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self.llm = llm

    async def verify_extraction_quality(
        self, paper_frameworks: list[dict[str, Any]], extraction_iteration: int
    ) -> VerificationFeedback:
        """Верифицирует качество извлеченных данных и генерирует фидбек.
        
        Args:
            paper_frameworks: Список извлеченных данных о фреймворках из статей
            extraction_iteration: Текущая итерация извлечения
            
        Returns:
            VerificationFeedback с оценкой качества и рекомендациями
        """
        if not self.llm:
            # Без LLM считаем качество достаточным
            return VerificationFeedback(
                quality_score=100,
                need_rerun=False,
                reasons=[],
                instructions_for_extraction="",
            )

        # Подсчитываем общее количество фреймворков
        total_frameworks = sum(len(paper_data.get("frameworks", [])) for paper_data in paper_frameworks)
        
        # Формируем сводку для анализа
        summary = []
        for paper_data in paper_frameworks:
            title = paper_data.get("title", "")
            frameworks_count = len(paper_data.get("frameworks", []))
            if frameworks_count > 0:
                summary.append(f"Статья '{title}': найдено {frameworks_count} фреймворк(ов)")
        
        summary_text = "\n".join(summary) if summary else "Фреймворки не найдены"

        system_prompt = """
        Ты ассистент-верификатор качества извлечения информации о AutoML-фреймворках из научных статей.
        
        Твоя задача:
        1. Оценить качество извлеченных данных (quality_score от 0 до 100)
        2. Определить, нужно ли повторить извлечение (need_rerun)
        3. Указать причины низкого качества (reasons)
        4. Дать конкретные инструкции для улучшения (instructions_for_extraction)
        
        Критерии оценки качества:
        - 80-100: Отличное качество - найдены все фреймворки, полные описания, есть примеры кода/бенчмарки
        - 60-79: Хорошее качество - найдены основные фреймворки, но не хватает деталей
        - 40-59: Среднее качество - найдены не все фреймворки или неполные описания
        - 20-39: Низкое качество - найдено мало фреймворков, много пропущено
        - 0-19: Очень низкое качество - фреймворки не найдены или найдены неправильно
        
        Когда нужно повторить извлечение (need_rerun = True):
        - quality_score < 70 И extraction_iteration < 3
        - Если из текста статьи явно видно, что должно быть больше фреймворков
        - Если описания фреймворков неполные (нет ключевых полей: description, categories, architecture)
        
        Когда НЕ нужно повторять (need_rerun = False):
        - quality_score >= 70
        - extraction_iteration >= 3 (достигнут лимит итераций)
        - Если из текста статьи реально больше ничего не выжать (статья не про AutoML-фреймворки)
        
        Всегда отвечай только в формате JSON, строго согласно заданной схеме.
        """

        human_prompt = f"""
        Проанализируй качество извлечения информации о AutoML-фреймворках.
        
        Текущая итерация: {extraction_iteration}/3
        
        Сводка извлеченных данных:
        {summary_text}
        
        Общее количество найденных фреймворков: {total_frameworks}
        
        Детали по каждой статье:
        """
        
        for idx, paper_data in enumerate(paper_frameworks, 1):
            title = paper_data.get("title", "")
            frameworks = paper_data.get("frameworks", [])
            human_prompt += f"\n\nСтатья {idx}: {title}\n"
            human_prompt += f"Найдено фреймворков: {len(frameworks)}\n"
            for fw in frameworks[:3]:  # Показываем первые 3 для анализа
                fw_name = fw.get("name", "Unknown")
                fw_desc = fw.get("description", "")[:200] if fw.get("description") else "Нет описания"
                fw_categories = fw.get("categories", [])
                fw_code = len(fw.get("code", []))
                human_prompt += f"  - {fw_name}: описание={len(fw_desc)} символов, категории={len(fw_categories)}, примеры кода={fw_code}\n"

        try:
            structured_llm = self.llm.with_structured_output(VerificationFeedback)
            result: VerificationFeedback = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
            
            # Логируем результат верификации
            logger.info(
                f"Верификация: score={result.quality_score}, "
                f"need_rerun={result.need_rerun}, "
                f"iteration={extraction_iteration}"
            )
            if result.reasons:
                logger.info(f"Причины: {', '.join(result.reasons)}")
            
            return result
        except Exception as e:
            logger.warning(f"Ошибка при верификации: {e}")
            # В случае ошибки считаем качество достаточным, чтобы не блокировать процесс
            return VerificationFeedback(
                quality_score=70,
                need_rerun=False,
                reasons=[f"Ошибка верификации: {str(e)}"],
                instructions_for_extraction="",
            )


class PaperSummarizer:
    """Класс для извлечения информации о фреймворках из научных статей."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self.llm = llm

    @staticmethod
    async def extract_text_from_pdf(pdf_url: str) -> str:
        """Извлечение текста из PDF файла."""
        if not pdf_url:
            return ""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()
                
                if len(response.content) == 0:
                    return ""
                
                reader = PdfReader(io.BytesIO(response.content))
                if len(reader.pages) == 0:
                    return ""
                
                text_parts = []
                for page in reader.pages:
                    try:
                        if page_text := page.extract_text():
                            text_parts.append(page_text)
                    except Exception:
                        continue
                
                return "\n".join(text_parts)
        except Exception:
            return ""

    async def extract_frameworks(self, text: str, paper_url: str = "", feedback: str | None = None) -> list[FrameworkInfo]:
        """Извлечение информации об AutoML фреймворках из текста.
        
        Args:
            text: Текст для анализа
            paper_url: URL статьи
            feedback: Обратная связь от предыдущей итерации верификации для улучшения извлечения
        """
        if not text or not self.llm:
            return []

        try:
            parser = PydanticOutputParser(pydantic_object=FrameworksList)
            
            feedback_section = ""
            if feedback:
                feedback_section = f"""
            Обратная связь от предыдущей итерации верификации:
            {feedback}
            
            Учти эти замечания при извлечении информации и улучши результат.
            Важно: не удаляй и не игнорируй уже корректно извлечённые фреймворки, если нет явного указания, что они ошибочны.
            Цель — улучшение полноты и качества, а не генерация совершенно нового набора.
            """
            
            prompt = f"""
            Ты — помощник по извлечению структурированных данных о AutoML-фреймворках из научного текста.

            Задача:
            1. Найти в тексте все упоминания AutoML-фреймворков, библиотек, платформ или систем,
            связанных с автоматизацией машинного обучения (подбор моделей, гиперпараметров,
            пайплайнов, NAS, автоматический feature engineering и т.п.).
            2. Для каждого найденного фреймворка сформировать объект FrameworkInfo.

            Заполни для КАЖДОГО фреймворка следующие поля FrameworkInfo:
            - name: название фреймворка (как указано в тексте, при необходимости слегка нормализуй).
            - description: краткое текстовое описание фреймворка, его назначения и ключевых возможностей.
            - strengths: список сильных сторон/преимуществ фреймворка (по одному пункту на элемент списка).
            - weaknesses: список слабых сторон/ограничений фреймворка.
            - categories: список категорий, например:
            "tabular", "time-series", "image", "text", "multimodal",
            "HPO", "NAS", "pipeline-automation", "LLM-based AutoML" и т.п.
            - architecture: список основных архитектурных компонентов/модулей фреймворка,
            например: "searcher", "evaluator", "meta-learner", "feature-engineering-module",
            "pipeline-generator", "orchestrator" и т.п.
            - benchmarks: список строк с описанием результатов в бенчмарках
            (на каких бенчмарках участвовал, какие места/метрики показал, какие датасеты использовались).
            Один бенчмарк или результат = один элемент списка.
            - code: список строк с примерами кода использования фреймворка
            (фрагменты API, примеры запуска, минимальные примеры из текста).
            Один пример кода = один элемент списка.
            - repository_urls: список ссылок на репозитории (GitHub, GitLab и другие ссылки на исходный код).
            Ищи упоминания GitHub, GitLab, URLs в тексте. Одна ссылка = один элемент списка.

            Важные правила:
            - Используй ТОЛЬКО информацию из текста. Ничего не выдумывай и не добавляй из внешних знаний.
            - Если какое-то поле явно не указано в тексте (например, нет примеров кода или архитектуры),
            оставь для него пустой список или пустую строку (в соответствии со схемой).
            - Если один и тот же фреймворк упомянут в нескольких местах, объединяй информацию в один объект
            FrameworkInfo (не дублируй одинаковые фреймворки с разными кусками описаний).
            - НЕ включай модели, которые являются просто ML/LLM-моделями (например, GPT-4o, GPT-4o-mini,
            Llama-3.3-70B, Meta-Llama-3.1-8B, Mistral-Nemo-2407, Phi-4, Qwen2.5-32B и т.п.), если они
            не описаны как часть AutoML-системы или AutoML-фреймворка.
{feedback_section}
            Текст для анализа:
{text}

            Ответ должен быть строго в формате, описанном ниже.
            Не добавляй никакого текста до или после JSON.

            {parser.get_format_instructions()}

            Если в тексте нет ни одного AutoML-фреймворка, верни пустой список фреймворков.
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, "content") else str(response)
            
            for attempt in range(2):
                try:
                    result = parser.parse(response_text)
                    if isinstance(result, FrameworksList):
                        for fw in result.frameworks:
                            if hasattr(fw, "paper_url") and not fw.paper_url:
                                fw.paper_url = paper_url
                        return result.frameworks
                except Exception as e:
                    if attempt == 0:
                        fix_prompt = f"""
                        Исправь JSON, чтобы он строго соответствовал формату.

                        Ошибка парсинга: {str(e)}

                        Исходный ответ модели:
{response_text}

Требуемый формат:
                        {parser.get_format_instructions()}

                        Верни только исправленный JSON без пояснений.
                        """
                        fix_response = await self.llm.ainvoke([HumanMessage(content=fix_prompt)])
                        response_text = fix_response.content if hasattr(fix_response, "content") else str(fix_response)
                    else:
                        return []

            return []
        except (AuthenticationError, ValueError, OSError, Exception):
            return []

    async def extract_frameworks_from_paper(
        self, paper_info: PaperInfo, extract_from_abstract: bool = True, feedback: str | None = None
    ) -> dict[str, Any]:
        """Извлечение информации о фреймворках из статьи.
        
        Args:
            paper_info: Информация о статье
            extract_from_abstract: Извлекать из абстракта или полного PDF
            feedback: Обратная связь от верификатора для улучшения извлечения
        """
        if extract_from_abstract:
            text = paper_info.abstract
        else:
            text = await self.extract_text_from_pdf(paper_info.pdf_url) if paper_info.pdf_url else ""
            text = text.strip() if text else paper_info.abstract

        frameworks = await self.extract_frameworks(text, paper_url=paper_info.url, feedback=feedback)

        return {
            "title": paper_info.title,
            "authors": paper_info.authors,
            "abstract": paper_info.abstract,
            "url": paper_info.url,
            "source": paper_info.source,
            "published_date": paper_info.published_date,
            "frameworks": [fw.model_dump() if hasattr(fw, "model_dump") else fw for fw in frameworks],
        }


def merge_frameworks(f1: dict[str, Any], f2: dict[str, Any]) -> dict[str, Any]:
    """Объединяет два фреймворка в один с четкой стратегией merge.
    
    Args:
        f1: Первый фреймворк (базовый)
        f2: Второй фреймворк (для объединения)
        
    Returns:
        Объединенный фреймворк
    """
    from collections import OrderedDict
    
    merged = f1.copy()
    
    # Списковые поля: set-объединение с сохранением порядка
    list_keys = [
        "strengths", "weaknesses", "categories", "architecture",
        "benchmarks", "code", "repository_urls"
    ]
    
    for key in list_keys:
        existing = merged.get(key, [])
        new = f2.get(key, [])
        # Объединяем и удаляем дубликаты, сохраняя порядок
        seen = OrderedDict()
        for item in existing + new:
            if item and item not in seen:
                seen[item] = None
        merged[key] = list(seen.keys())
    
    # Описание: берем самое длинное или объединяем
    desc1 = merged.get("description", "")
    desc2 = f2.get("description", "")
    if len(desc2) > len(desc1):
        merged["description"] = desc2
    elif desc2 and desc2 not in desc1:
        merged["description"] = f"{desc1}\n\nДополнение: {desc2}"
    
    # paper_urls: объединяем все уникальные URL
    paper_urls = merged.setdefault("paper_urls", [])
    for url in f2.get("paper_urls", []):
        if url and url not in paper_urls:
            paper_urls.append(url)
    
    # name и slug: используем из базового (f1)
    # Если slug отсутствует, генерируем из name
    if not merged.get("slug") and merged.get("name"):
        from .models import generate_slug
        merged["slug"] = generate_slug(merged["name"])
    
    return merged


class FrameworkEnricher:
    """Класс для обогащения данных о фреймворках через Tavily API."""

    def __init__(self, llm: ChatOpenAI | None = None, tavily_api_key: str | None = None) -> None:
        self.llm = llm
        self._tavily_client: TavilyClient | None = None
        self._cache: dict[str, Any] = {}  # Кэш для запросов Tavily
        # Задержка между запросами (секунды) - можно настроить через TAVILY_RATE_LIMIT_DELAY
        self._rate_limit_delay = float(os.getenv("TAVILY_RATE_LIMIT_DELAY", "2.0"))
        self._rate_limit_exceeded = False  # Флаг превышения лимита
        self._last_request_time = 0.0  # Время последнего запроса
        
        if TAVILY_AVAILABLE and tavily_api_key:
            try:
                self._tavily_client = TavilyClient(api_key=tavily_api_key)
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Tavily клиент: {e}")
        elif not TAVILY_AVAILABLE:
            logger.warning("Tavily не доступен. Установите tavily-python для обогащения данных.")

    async def search_tavily(self, query: str, max_results: int = 5, max_retries: int = 3) -> list[dict[str, Any]]:
        """Поиск информации через Tavily API с retry логикой и rate limiting.
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            max_retries: Максимальное количество попыток при ошибках
            
        Returns:
            Список результатов поиска
        """
        if not self._tavily_client:
            return []
        
        # Если превышен лимит, пропускаем запрос
        if self._rate_limit_exceeded:
            logger.debug(f"Tavily rate limit превышен, пропускаем запрос: {query[:50]}")
            return []
        
        # Проверяем кэш
        cache_key = f"{query}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Rate limiting: задержка между запросами
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last_request)
        
        # Retry логика только для временных сетевых ошибок (не для SSL)
        last_error = None
        for attempt in range(max_retries):
            try:
                # Обновляем время последнего запроса
                self._last_request_time = time.time()
                
                # Используем executor для синхронного вызова
                def _search():
                    response = self._tavily_client.search(query=query, max_results=max_results)
                    return response.get("results", [])
                
                results = await asyncio.get_event_loop().run_in_executor(None, _search)
                self._cache[cache_key] = results
                if attempt > 0:
                    logger.info(f"Tavily поиск успешен после {attempt + 1} попыток для запроса: {query[:50]}")
                
                # Задержка после успешного запроса для rate limiting
                await asyncio.sleep(self._rate_limit_delay)
                return results
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Проверяем тип ошибки
                is_ssl_error = "SSL" in error_msg or "SSLError" in error_msg or "SSL: UNEXPECTED_EOF" in error_msg
                is_connection_error = "Connection" in error_msg or "timeout" in error_msg.lower() or "Max retries" in error_msg
                is_rate_limit_error = (
                    "excessive requests" in error_msg.lower() or 
                    "usage limit" in error_msg.lower() or
                    "rate limit" in error_msg.lower()
                )
                
                if is_rate_limit_error:
                    # Превышен лимит запросов - устанавливаем флаг и прекращаем попытки
                    self._rate_limit_exceeded = True
                    logger.warning(
                        f"⚠️  Tavily rate limit превышен для запроса: {query[:50]}\n"
                        f"   Обогащение через Tavily будет пропущено для оставшихся фреймворков.\n"
                        f"   Рекомендации:\n"
                        f"   - Увеличьте задержку между запросами (rate_limit_delay)\n"
                        f"   - Обновите план Tavily для увеличения лимита\n"
                        f"   - Обработайте фреймворки батчами с большими паузами"
                    )
                    return []  # Не делаем retry для rate limit ошибок
                elif is_ssl_error:
                    # SSL ошибки обычно не решаются retry - это проблема конфигурации/сети
                    logger.error(
                        f"SSL ошибка при поиске в Tavily: {error_msg[:200]}\n"
                        f"   Это обычно указывает на проблему с SSL/TLS соединением.\n"
                        f"   Возможные решения:\n"
                        f"   - Проверьте настройки прокси/файрвола\n"
                        f"   - Обновите SSL сертификаты\n"
                        f"   - Проверьте версию Python и библиотек\n"
                        f"   - Попробуйте позже (возможно, временная проблема на стороне сервера)"
                    )
                    return []  # Не делаем retry для SSL ошибок
                elif is_connection_error and attempt < max_retries - 1:
                    # Для временных сетевых ошибок делаем retry с экспоненциальной задержкой
                    wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2s, 4s, 6s
                    logger.warning(
                        f"Ошибка соединения с Tavily (попытка {attempt + 1}/{max_retries}): {error_msg[:100]}"
                    )
                    logger.info(f"Повторная попытка через {wait_time} секунд...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Для других ошибок или если превышен лимит попыток
                    if is_connection_error:
                        logger.error(f"Ошибка соединения с Tavily после {max_retries} попыток: {error_msg[:200]}")
                    else:
                        logger.warning(f"Ошибка при поиске в Tavily: {error_msg[:200]}")
                    break
        
        # Если все попытки неудачны
        if last_error:
            logger.error(f"Не удалось выполнить поиск в Tavily после {max_retries} попыток. Последняя ошибка: {last_error}")
        return []

    async def enrich_framework(self, framework: dict[str, Any]) -> dict[str, Any]:
        """Обогащает данные о фреймворке информацией из интернета.
        
        Args:
            framework: Словарь с данными о фреймворке
            
        Returns:
            Обогащенный фреймворк
        """
        if not self.llm or not self._tavily_client:
            return framework
        
        name = framework.get("name", "")
        if not name:
            return framework
        
        logger.info(f"Обогащение данных для фреймворка: {name}")
        
        # Стратегия запросов
        queries = [
            f"{name} AutoML GitHub repository",
            f"{name} documentation site:readthedocs.io OR site:docs.* OR site:github.io",
            f"{name} benchmarks tabular",
            f"{name} examples code",
        ]
        
        all_results = []
        successful_queries = 0
        failed_queries = 0
        
        for query in queries[:3]:  # Максимум 3 запроса на фреймворк
            results = await self.search_tavily(query, max_results=3)
            if results:
                all_results.extend(results)
                successful_queries += 1
            else:
                failed_queries += 1
            if len(all_results) >= 10:  # Ограничение по количеству результатов
                break
        
        if not all_results:
            if failed_queries > 0:
                logger.warning(
                    f"Tavily не смог выполнить поиск для {name}: "
                    f"{failed_queries} запросов завершились с ошибками"
                )
            else:
                logger.info(f"Tavily не нашел результатов поиска для {name}")
            logger.debug(f"Запросы: {queries[:3]}")
            return framework
        
        logger.info(
            f"Tavily нашел {len(all_results)} результатов для {name} "
            f"(успешных запросов: {successful_queries}, неудачных: {failed_queries})"
        )
        
        # Извлекаем информацию через LLM
        try:
            enriched_data = await self._extract_enrichment_data(name, all_results, framework)
            # Объединяем с существующими данными (только дополняем, не перезаписываем)
            return self._merge_enrichment(framework, enriched_data)
        except Exception as e:
            logger.warning(f"Ошибка при обогащении {name}: {e}")
            return framework

    async def _extract_enrichment_data(
        self, name: str, search_results: list[dict[str, Any]], existing: dict[str, Any]
    ) -> dict[str, Any]:
        """Извлекает структурированные данные из результатов поиска через LLM."""
        if not self.llm:
            return {}
        
        # Формируем текст из результатов поиска
        results_text = "\n\n".join([
            f"URL: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')[:500]}"
            for r in search_results[:5]
        ])
        
        parser = PydanticOutputParser(pydantic_object=FrameworksList)
        prompt = f"""
        Ты помощник по извлечению дополнительной информации о AutoML-фреймворке из результатов веб-поиска.
        
        Задача: Найти и извлечь дополнительную информацию о фреймворке "{name}".
        
        Важные правила:
        - НЕ изменяй существующие данные, только ДОПОЛНЯЙ новыми фактами
        - Если информация противоречит тексту статьи — НЕ переписывай оригинальное описание
        - Чётко указать структуру: что именно дополняем (code, benchmarks, repository_urls)
        - ЗАПРЕЩЕНО придумывать репозитории/бенчмарки, если их нет в данных поиска
        
        Извлекай ТОЛЬКО:
        - Примеры кода (code) - фрагменты API, примеры использования
        - Бенчмарки (benchmarks) - результаты на датасетах, метрики
        - Ссылки на репозитории (repository_urls) - GitHub, GitLab и т.п.
        
        Существующие данные о фреймворке:
        - Описание: {existing.get('description', '')[:200]}
        - Репозитории: {existing.get('repository_urls', [])}
        
        Результаты поиска:
        {results_text}
        
        Верни структурированные данные в формате FrameworksList (один фреймворк).
        Если дополнительной информации нет, верни пустой список фреймворков.
        
        {parser.get_format_instructions()}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, "content") else str(response)
            
            result = parser.parse(response_text)
            if isinstance(result, FrameworksList) and result.frameworks:
                extracted = result.frameworks[0].model_dump()
                # Логируем, что было извлечено
                repos_count = len(extracted.get('repository_urls', []))
                code_count = len(extracted.get('code', []))
                benchmarks_count = len(extracted.get('benchmarks', []))
                if repos_count > 0 or code_count > 0 or benchmarks_count > 0:
                    logger.info(
                        f"LLM извлек из результатов Tavily для {name}: "
                        f"репозитории={repos_count}, код={code_count}, бенчмарки={benchmarks_count}"
                    )
                else:
                    logger.info(f"LLM не извлек данных из результатов Tavily для {name} (возможно, данные уже есть или не найдены)")
                return extracted
            else:
                logger.info(f"LLM вернул пустой результат для {name}")
            return {}
        except Exception as e:
            logger.warning(f"Ошибка при парсинге обогащенных данных для {name}: {e}")
            return {}

    def _merge_enrichment(self, existing: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, Any]:
        """Объединяет обогащенные данные с существующими (только дополняет)."""
        merged = existing.copy()
        
        # Дополняем списковые поля
        added_count = 0
        for key in ["code", "benchmarks", "repository_urls"]:
            existing_list = merged.get(key, [])
            new_list = enrichment.get(key, [])
            before_count = len(existing_list)
            # Добавляем только новые элементы
            for item in new_list:
                if item and item not in existing_list:
                    existing_list.append(item)
                    added_count += 1
            merged[key] = existing_list
            after_count = len(existing_list)
            if after_count > before_count:
                logger.debug(f"Добавлено {after_count - before_count} элементов в поле '{key}'")
        
        if added_count == 0 and any(enrichment.get(key) for key in ["code", "benchmarks", "repository_urls"]):
            name = existing.get("name", "Unknown")
            logger.info(f"Данные для {name} не добавлены: все элементы уже присутствуют (дубликаты)")
        
        return merged


class PaperResearchAgent:
    """Агент для поиска и извлечения информации о AutoML фреймворках из научных статей с использованием LangGraph."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        test_mode: bool = False,
    ) -> None:
        """Инициализация агента.
        
        Args:
            llm_config: Конфигурация LLM для этого агента. OpenRouter всегда берется из .env,
                        но модель можно указать для каждого агента отдельно.
                        Пример: LLMConfig(model="anthropic/claude-3-opus")
                        Если не указана, используется дефолтная модель из settings.agents.
            test_mode:  Если True, режим для быстрого тестирования:
                        - Ищет меньше статей (2 вместо 6)
                        - Извлекает информацию только из абстрактов, а не из полного PDF
        """
        self.name = "paper_researcher"
        self._test_mode = test_mode
        self._search_tools = PaperSearchTools()
        
        # Инициализируем LLM в агенте (только если не в тестовом режиме)
        if not test_mode:
            llm = self._create_llm(llm_config)
            self._analyzer = PaperAnalyzer(llm=llm)
            self._summarizer = PaperSummarizer(llm=llm)
            self._verifier = FrameworkVerifier(llm=llm)
            # Инициализируем enricher с Tavily API key из окружения
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            self._enricher = FrameworkEnricher(llm=llm, tavily_api_key=tavily_api_key)
        else:
            self._analyzer = None
            self._summarizer = None
            self._verifier = None
            self._enricher = None
        self._graph = self._build_graph()

    @staticmethod
    def _create_llm(llm_config: LLMConfig | None = None) -> ChatOpenAI:
        """Создает экземпляр ChatOpenAI из конфигурации."""
        if llm_config is None:
            llm_config = getattr(settings.agents, "paper_researcher", None) or LLMConfig.from_env()

        api_key = llm_config.api_key or os.getenv("MAS_LLM__API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key не найден. Установите OPENROUTER_API_KEY в .env")

        base_url = (
            llm_config.base_url
            or os.getenv("BASE_URL")
            or os.getenv("MAS_LLM__BASE_URL")
            or "https://openrouter.ai/api/v1"
        )

        return ChatOpenAI(model=llm_config.model, api_key=api_key, base_url=base_url)

    def _build_graph(self):
        """Построение графа состояний."""
        workflow = StateGraph(PaperSearchState)
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("analyze_papers", self._analyze_papers_node)
        workflow.add_node("extract_frameworks", self._extract_frameworks_node)
        workflow.add_node("verify_frameworks", self._verify_frameworks_node)
        workflow.add_node("merge_frameworks", self._merge_frameworks_node)
        workflow.add_node("deduplicate_frameworks", self._deduplicate_frameworks_node)
        workflow.add_node("enrich_frameworks", self._enrich_frameworks_node)
        workflow.set_entry_point("search_papers")
        workflow.add_edge("search_papers", "analyze_papers")
        workflow.add_edge("analyze_papers", "extract_frameworks")
        workflow.add_edge("extract_frameworks", "verify_frameworks")
        workflow.add_conditional_edges(
            "verify_frameworks",
            self._should_rerun_extraction,
            {
                "rerun": "extract_frameworks",
                "continue": "merge_frameworks",
                "end": END,
            },
        )
        workflow.add_edge("merge_frameworks", "deduplicate_frameworks")
        workflow.add_edge("deduplicate_frameworks", "enrich_frameworks")
        workflow.add_edge("enrich_frameworks", END)
        return workflow.compile()

    def _should_rerun_extraction(self, state: PaperSearchState) -> str:
        """Определяет, нужно ли повторить извлечение или продолжить.
        
        Returns:
            "rerun" - если нужно повторить извлечение
            "continue" - если продолжить к merge_frameworks
            "end" - если завершить (ранний выход)
        """
        # Получаем последний результат верификации из состояния
        # (он был сохранен в _verify_frameworks_node)
        need_rerun = state.get("verification_need_rerun", False)
        extraction_iteration = state.get("extraction_iteration", 0)
        paper_frameworks = state.get("paper_frameworks", [])
        
        # Ранний выход: если нет фреймворков и не нужно повторять
        if not paper_frameworks and not need_rerun:
            logger.info("Ранний выход: фреймворки не найдены и повтор не требуется")
            return "end"
        
        # Повторяем, если нужно и не достигнут лимит
        if need_rerun and extraction_iteration < 3:
            logger.info(f"Повтор извлечения: итерация {extraction_iteration + 1}/3")
            return "rerun"
        
        # Продолжаем к merge
        return "continue"

    async def _search_papers_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для поиска статей."""
        max_results = state.get("max_results") or (2 if self._test_mode else 6)
        papers = await self._search_tools.search_arxiv(
            state["query"],
            max_results=max_results,
            sort_by=state.get("sort_by"),
            days_back=state.get("days_back"),
        )

        return {
            **state,
            "papers": [
            {
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "url": p.url,
                "source": p.source,
                "pdf_url": p.pdf_url,
                "published_date": p.published_date,
            }
                for p in papers
            ],
            "filtered_papers": [],
            "paper_frameworks": [],
            "frameworks": [],
            "extraction_iteration": 0,
            "verification_feedback": None,
            "verification_score": None,
        }

    async def _analyze_papers_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для анализа и фильтрации статей по релевантности."""
        if not self._analyzer:
            # Без анализатора пропускаем все статьи
            return {**state, "filtered_papers": state["papers"]}

        filtered_papers = []

        for paper_dict in state["papers"]:
            paper_info = PaperInfo(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                abstract=paper_dict["abstract"],
                url=paper_dict["url"],
                source=paper_dict["source"],
                pdf_url=paper_dict.get("pdf_url"),
                published_date=paper_dict.get("published_date"),
            )

            is_relevant, reason = await self._analyzer.analyze_paper_relevance(paper_info)

            if is_relevant:
                filtered_papers.append(paper_dict)
                logger.info(f"Статья принята: '{paper_info.title}'")
            else:
                logger.info(
                    f"Статья отклонена: '{paper_info.title}' | "
                    f"URL: {paper_info.url} | "
                    f"Причина: {reason}"
                )

        logger.info(f"Отфильтровано статей: {len(state['papers'])} → {len(filtered_papers)}")

        return {**state, "filtered_papers": filtered_papers}

    async def _extract_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для извлечения информации о фреймворках из статей."""
        if not self._summarizer:
            return {**state, "paper_frameworks": []}

        # Получаем текущую итерацию и фидбек
        extraction_iteration = state.get("extraction_iteration", 0)
        verification_feedback = state.get("verification_feedback")
        
        logger.info(f"Извлечение фреймворков: итерация {extraction_iteration + 1}/3")
        if verification_feedback:
            logger.info(f"Используется фидбек от верификатора")

        paper_frameworks = []
        # Используем отфильтрованные статьи
        for paper_dict in state.get("filtered_papers", state.get("papers", [])):
            paper_info = PaperInfo(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                abstract=paper_dict["abstract"],
                url=paper_dict["url"],
                source=paper_dict["source"],
                pdf_url=paper_dict.get("pdf_url"),
                published_date=paper_dict.get("published_date"),
            )
            paper_data = await self._summarizer.extract_frameworks_from_paper(
                paper_info, extract_from_abstract=self._test_mode, feedback=verification_feedback
            )
            paper_frameworks.append(paper_data)
        
        # Инкрементируем итерацию
        new_iteration = extraction_iteration + 1
        logger.info(f"Извлечено фреймворков из {len(paper_frameworks)} статей (итерация {new_iteration})")
            
        return {
            **state,
            "paper_frameworks": paper_frameworks,
            "extraction_iteration": new_iteration,
        }

    async def _verify_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для верификации качества извлеченных данных о фреймворках."""
        if not self._verifier:
            # Без верификатора пропускаем верификацию
            return {
                **state,
                "verification_feedback": None,
                "verification_score": None,
            }

        extraction_iteration = state.get("extraction_iteration", 0)
        paper_frameworks = state.get("paper_frameworks", [])
        
        # Верифицируем качество извлечения
        feedback = await self._verifier.verify_extraction_quality(
            paper_frameworks, extraction_iteration
        )
        
        # Сохраняем в состояние только текстовый фидбек и опционально score
        # need_rerun используется локально в условной функции
        return {
            **state,
            "verification_feedback": feedback.instructions_for_extraction or None,
            "verification_score": feedback.quality_score,
            "verification_need_rerun": feedback.need_rerun,  # Временно для условной функции
        }

    async def _merge_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для объединения информации о фреймворках из разных статей."""
        frameworks_dict: dict[str, dict[str, Any]] = {}

        for paper_data in state.get("paper_frameworks", []):
            paper_url = paper_data.get("url", "")
            for fw_data in paper_data.get("frameworks", []):
                fw_dict = fw_data if isinstance(fw_data, dict) else fw_data.model_dump()
                name = fw_dict.get("name", "")
                if not name:
                    continue
                
                # Добавляем paper_url
                fw_dict = fw_dict.copy()
                fw_dict.setdefault("paper_urls", [])
                if paper_url and paper_url not in fw_dict["paper_urls"]:
                    fw_dict["paper_urls"].append(paper_url)
                if "paper_url" not in fw_dict:
                    fw_dict["paper_url"] = paper_url
                
                if name in frameworks_dict:
                    # Используем функцию merge_frameworks для объединения
                    frameworks_dict[name] = merge_frameworks(frameworks_dict[name], fw_dict)
                else:
                    frameworks_dict[name] = fw_dict

        frameworks_count = len(frameworks_dict)
        logger.info(f"Объединено фреймворков из разных статей: {frameworks_count}")
        return {**state, "frameworks": list(frameworks_dict.values())}

    async def _deduplicate_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для дедупликации фреймворков по slug и похожим названиям."""
        frameworks = state.get("frameworks", [])
        if not frameworks:
            return state
        
        from .models import generate_slug
        
        # Группируем по slug
        frameworks_by_slug: dict[str, list[dict[str, Any]]] = {}
        for fw in frameworks:
            # Генерируем slug, если отсутствует
            slug = fw.get("slug") or generate_slug(fw.get("name", ""))
            fw["slug"] = slug  # Обновляем slug в фреймворке
            
            if slug not in frameworks_by_slug:
                frameworks_by_slug[slug] = []
            frameworks_by_slug[slug].append(fw)
        
        # Объединяем дубликаты
        deduplicated: list[dict[str, Any]] = []
        duplicates_found = 0
        
        for slug, fw_list in frameworks_by_slug.items():
            if len(fw_list) > 1:
                # Найдены дубликаты - объединяем
                duplicates_found += len(fw_list) - 1
                logger.info(f"Найдены дубликаты для slug '{slug}': {len(fw_list)} фреймворков")
                
                # Объединяем все в один
                merged = fw_list[0]
                for fw in fw_list[1:]:
                    merged = merge_frameworks(merged, fw)
                deduplicated.append(merged)
            else:
                deduplicated.append(fw_list[0])
        
        if duplicates_found > 0:
            logger.info(f"Объединено {duplicates_found} дубликатов. Итого фреймворков: {len(deduplicated)}")
        else:
            logger.info(f"Дубликаты не найдены. Фреймворков: {len(deduplicated)}")
        
        return {**state, "frameworks": deduplicated}

    async def _enrich_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для обогащения данных о фреймворках через Tavily."""
        if not self._enricher:
            logger.info("Enricher не доступен, пропускаем обогащение")
            logger.info("Причина: Tavily клиент не инициализирован (проверьте TAVILY_API_KEY)")
            return state
        
        # Проверяем, есть ли Tavily клиент
        if not hasattr(self._enricher, '_tavily_client') or not self._enricher._tavily_client:
            logger.info("Tavily клиент не доступен, пропускаем обогащение")
            logger.info("Причина: Tavily API ключ не установлен или клиент не инициализирован")
            return state
        
        frameworks = state.get("frameworks", [])
        if not frameworks:
            return state
        
        logger.info(f"Начало обогащения данных для {len(frameworks)} фреймворков")
        enriched_frameworks = []
        enriched_count = 0
        skipped_count = 0
        no_results_count = 0
        
        for idx, fw in enumerate(frameworks, 1):
            try:
                name = fw.get("name", "Unknown")
                
                # Проверяем, не превышен ли rate limit
                if hasattr(self._enricher, '_rate_limit_exceeded') and self._enricher._rate_limit_exceeded:
                    logger.info(
                        f"Пропуск обогащения для {name} ({idx}/{len(frameworks)}): "
                        f"Tavily rate limit превышен"
                    )
                    enriched_frameworks.append(fw)  # Оставляем оригинальный без обогащения
                    skipped_count += 1
                    continue
                
                enriched = await self._enricher.enrich_framework(fw)
                
                # Проверяем, были ли найдены результаты поиска
                # (это можно определить по тому, изменились ли данные)
                repos_before = len(fw.get("repository_urls", []))
                code_before = len(fw.get("code", []))
                benchmarks_before = len(fw.get("benchmarks", []))
                
                repos_after = len(enriched.get("repository_urls", []))
                code_after = len(enriched.get("code", []))
                benchmarks_after = len(enriched.get("benchmarks", []))
                
                if enriched != fw:
                    enriched_count += 1
                    logger.info(
                        f"Обогащен {name} ({idx}/{len(frameworks)}): "
                        f"репозитории {repos_before}→{repos_after}, "
                        f"код {code_before}→{code_after}, "
                        f"бенчмарки {benchmarks_before}→{benchmarks_after}"
                    )
                else:
                    # Данные не изменились - возможно, Tavily не нашел результатов
                    no_results_count += 1
                    logger.debug(f"Tavily не нашел дополнительных данных для {name} ({idx}/{len(frameworks)})")
                
                enriched_frameworks.append(enriched)
            except Exception as e:
                logger.warning(f"Ошибка при обогащении фреймворка '{fw.get('name', 'Unknown')}' ({idx}/{len(frameworks)}): {e}")
                enriched_frameworks.append(fw)  # Оставляем оригинальный
                skipped_count += 1
        
        logger.info(
            f"Обогащение завершено: обогащено {enriched_count}, "
            f"без новых данных {no_results_count}, "
            f"пропущено {skipped_count}, всего {len(enriched_frameworks)}"
        )
        
        return {**state, "frameworks": enriched_frameworks}

    async def handle(
        self,
        query: str,
        sort_by: arxiv.SortCriterion | None = None,
        days_back: int | None = None,
        max_results: int | None = None,
    ) -> PaperSearchState:
        """Обработка запроса на поиск и извлечение информации о фреймворках."""
        if not query:
            return {
                "query": "",
                "papers": [],
                "filtered_papers": [],
                "paper_frameworks": [],
                "error": "Не указан запрос для поиска статей.",
                "sort_by": None,
                "days_back": None,
                "max_results": None,
                "frameworks": [],
                "extraction_iteration": 0,
                "verification_feedback": None,
                "verification_score": None,
            }

        try:
            initial_state: PaperSearchState = {
                "query": query,
                "papers": [],
                "filtered_papers": [],
                "paper_frameworks": [],
                "error": None,
                "sort_by": sort_by,
                "days_back": days_back,
                "max_results": max_results,
                "frameworks": [],
                "extraction_iteration": 0,
                "verification_feedback": None,
                "verification_score": None,
            }
            return await self._graph.ainvoke(initial_state)
        except (ValueError, OSError, KeyError) as e:
            return {
                "query": query,
                "papers": [],
                "filtered_papers": [],
                "paper_frameworks": [],
                "error": f"Ошибка при поиске статей: {str(e)}",
                "sort_by": sort_by,
                "days_back": days_back,
                "max_results": max_results,
                "frameworks": [],
                "extraction_iteration": 0,
                "verification_feedback": None,
                "verification_score": None,
            }

