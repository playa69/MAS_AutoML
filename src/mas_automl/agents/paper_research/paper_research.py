"""Агент для поиска и саммаризации научных статей с использованием LangGraph и LangChain."""

from __future__ import annotations

import asyncio
import io
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict
from urllib.error import URLError

import arxiv
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# В LangChain 1.0+ OutputFixingParser был удален из langchain_core
# Используем только PydanticOutputParser с ручной обработкой ошибок через повторные попытки
# Это работает так же эффективно, как OutputFixingParser
OutputFixingParser = None
from openai import AuthenticationError
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ...config.settings import LLMConfig, settings


class PaperSearchState(TypedDict):
    """Состояние для графа поиска и извлечения информации о фреймворках."""

    query: str
    papers: list[dict[str, Any]]
    paper_frameworks: list[dict[str, Any]]  # Информация о фреймворках из каждой статьи
    current_paper_index: int
    error: str | None
    sort_by: arxiv.SortCriterion | None
    days_back: int | None  # Фильтр по дате: только статьи за последние N дней
    max_results: int | None  # Максимальное количество статей для поиска
    frameworks: list[dict[str, Any]]  # Объединенная информация о всех найденных фреймворках


@dataclass
class PaperInfo:
    """Информация о научной статье."""

    title: str
    authors: list[str]
    abstract: str
    url: str
    source: Literal["arxiv"]
    pdf_url: str | None = None
    published_date: str | None = None


class FrameworkInfo(BaseModel):
    """Детальная информация об AutoML фреймворке."""

    name: str = Field(description="Название фреймворка")
    description: str = Field(
        default="",
        description="Описание фреймворка, его назначение и основные возможности"
    )
    advantages: list[str] = Field(
        default_factory=list,
        description="Список преимуществ фреймворка"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Сильные стороны фреймворка"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Слабые стороны и ограничения фреймворка"
    )
    supported_data_types: list[str] = Field(
        default_factory=list,
        description="Типы данных, с которыми работает фреймворк (tabular, image, text, multimodal и т.д.)"
    )
    unsupported_features: list[str] = Field(
        default_factory=list,
        description="Что фреймворк не поддерживает или ограничения в функциональности"
    )
    benchmark_results: list[str] = Field(
        default_factory=list,
        description="Результаты в бенчмарках: места, метрики, типы датасетов"
    )
    code_snippets: list[str] = Field(
        default_factory=list,
        description="Код из статьи, связанный с фреймворком (примеры использования, API и т.д.)"
    )
    repository_urls: list[str] = Field(
        default_factory=list,
        description="Ссылки на репозитории (GitHub, GitLab и т.д.)"
    )
    paper_url: str = Field(
        default="",
        description="URL статьи, из которой извлечена информация"
    )


class FrameworksList(BaseModel):
    """Модель для списка AutoML фреймворков с детальной информацией."""

    frameworks: list[FrameworkInfo] = Field(
        description="Список AutoML фреймворков с детальной информацией о каждом"
    )


class PaperSearchTools:
    """Инструменты для поиска научных статей."""

    @staticmethod
    async def search_arxiv(
        query: str,
        max_results: int = 5,
        sort_by: arxiv.SortCriterion | None = None,
        days_back: int | None = None,
    ) -> list[PaperInfo]:
        """Поиск статей в arXiv.
        
        Args:
            query: Запрос для поиска статей
            max_results: Максимальное количество результатов
            sort_by: Критерий сортировки (Relevance, LastUpdatedDate, SubmittedDate).
                     По умолчанию Relevance.
            days_back: Если указано, возвращает только статьи, опубликованные за последние N дней.
                      Используется вместе с sort_by=SubmittedDate для поиска новых статей.
        """
        try:
            # Определяем критерий сортировки
            if sort_by is None:
                sort_by = arxiv.SortCriterion.Relevance
            
            # Вычисляем дату начала периода для фильтрации, если указан days_back
            cutoff_date = None
            if days_back is not None:
                cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Увеличиваем max_results, если нужна фильтрация по дате,
            # чтобы после фильтрации осталось достаточно результатов
            search_max_results = max_results * 3 if days_back is not None else max_results
            
            # arxiv API синхронный, поэтому выполняем в executor
            def _search():
                search = arxiv.Search(
                    query=query,
                    max_results=search_max_results,
                    sort_by=sort_by,
                )
                return list(search.results())

            papers = await asyncio.get_event_loop().run_in_executor(None, _search)
            results = []
            for paper in papers:
                # Фильтруем по дате, если указан days_back
                if cutoff_date is not None:
                    if paper.published:
                        # Преобразуем дату публикации в datetime без timezone для сравнения
                        paper_date = paper.published
                        if paper_date.tzinfo is not None:
                            paper_date = paper_date.replace(tzinfo=None)
                        if paper_date < cutoff_date:
                            continue  # Пропускаем статьи старше cutoff_date
                    else:
                        continue  # Пропускаем статьи без даты публикации
                
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
                
                # Останавливаемся, когда набрали нужное количество результатов
                if len(results) >= max_results:
                    break
            
            return results
        except (URLError, ValueError, OSError, Exception) as e:
            print(f"Ошибка при поиске в arXiv: {e}")
            return []


class PaperSummarizer:
    """Класс для саммаризации научных статей."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        """Инициализация саммаризатора.
        
        Args:
            llm: Готовый экземпляр ChatOpenAI для саммаризации. Может быть None для fallback режима.
        """
        self.llm = llm

    @staticmethod
    def extract_frameworks_fallback(_text: str, _paper_url: str = "") -> list[FrameworkInfo]:
        """Fallback метод для извлечения фреймворков (без LLM).
        
        Args:
            _text: Текст для анализа (не используется в fallback)
            _paper_url: URL статьи (не используется в fallback)
            
        Returns:
            Пустой список, так как без LLM извлечение ненадежно
        """
        return []

    async def summarize_abstract(self, paper_text: str) -> str:
        """Саммаризация всей статьи (не только абстракта).
        
        Args:
            paper_text: Полный текст статьи (из PDF или абстракт, если PDF недоступен)
            
        Returns:
            Краткое резюме статьи на русском языке
        """
        if not paper_text:
            return "Текст статьи отсутствует."

        # Если нет LLM, возвращаем первые 500 символов текста
        if self.llm is None:
            return paper_text[:500] + "..." if len(paper_text) > 500 else paper_text

        try:
            prompt = f"""Создай краткое резюме следующей научной статьи на русском языке.
            Выдели основные идеи, методы, результаты и выводы.
            Резюме должно быть структурированным и информативным.

            Текст статьи:
            {paper_text}

            Резюме:"""

            message = HumanMessage(content=prompt)
            response = await self.llm.ainvoke([message])
            return response.content if hasattr(response, "content") else str(response)
        except AuthenticationError as e:
            error_msg = (
                f"Ошибка аутентификации API: {str(e)}\n"
                "Проверьте правильность API ключа в переменных окружения:\n"
                "- OPENROUTER_API_KEY или MAS_LLM__API_KEY\n"
                "Убедитесь, что используете правильный ключ для OpenRouter."
            )
            print(error_msg)
            return f"Ошибка саммаризации: {error_msg}"
        except (ValueError, OSError) as e:
            print(f"Ошибка при саммаризации статьи: {e}")
            return f"Ошибка саммаризации: {str(e)}"

    @staticmethod
    async def extract_text_from_pdf(pdf_url: str) -> str:
        """Извлечение текста из PDF файла.
        
        Args:
            pdf_url: URL PDF файла
            
        Returns:
            Извлеченный текст из PDF
        """
        if not pdf_url:
            print("[DEBUG] PDF URL is empty")
            return ""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"[DEBUG] Downloading PDF from: {pdf_url}")
                response = await client.get(pdf_url)
                response.raise_for_status()
                
                # Проверяем, что получили PDF
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
                    print(f"[WARNING] Content-Type is not PDF: {content_type}")
                
                # Проверяем размер файла
                content_length = len(response.content)
                print(f"[DEBUG] PDF file size: {content_length} bytes")
                if content_length == 0:
                    print("[ERROR] PDF file is empty")
                    return ""
                
                # Читаем PDF из байтов
                pdf_file = io.BytesIO(response.content)
                reader = PdfReader(pdf_file)
                
                # Проверяем количество страниц
                num_pages = len(reader.pages)
                print(f"[DEBUG] PDF has {num_pages} pages")
                if num_pages == 0:
                    print("[ERROR] PDF has no pages")
                    return ""
                
                # Извлекаем текст со всех страниц
                text_parts = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        else:
                            print(f"[WARNING] Page {i+1} has no extractable text (may be image-based)")
                    except Exception as page_error:
                        print(f"[WARNING] Error extracting text from page {i+1}: {page_error}")
                        continue
                
                extracted_text = "\n".join(text_parts)
                print(f"[DEBUG] Successfully extracted {len(extracted_text)} characters from PDF")
                
                if not extracted_text.strip():
                    print("[WARNING] Extracted text is empty - PDF may be image-based or encrypted")
                
                return extracted_text
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] HTTP error when downloading PDF: {e.response.status_code} {e.response.reason_phrase}")
            print(f"[ERROR] URL: {pdf_url}")
            return ""
        except httpx.TimeoutException as e:
            print(f"[ERROR] Timeout when downloading PDF: {e}")
            print(f"[ERROR] URL: {pdf_url}")
            return ""
        except httpx.RequestError as e:
            print(f"[ERROR] Request error when downloading PDF: {type(e).__name__}: {e}")
            print(f"[ERROR] URL: {pdf_url}")
            return ""
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else repr(e)
            print(f"[ERROR] Error extracting text from PDF: {error_type}: {error_msg}")
            print(f"[ERROR] URL: {pdf_url}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return ""

    async def extract_frameworks(self, text: str, paper_url: str = "") -> list[FrameworkInfo]:
        """Извлечение детальной информации об AutoML фреймворках из текста.
        
        Args:
            text: Текст для анализа (абстракт, саммари или полный текст статьи)
            paper_url: URL статьи для привязки информации
            
        Returns:
            Список найденных AutoML фреймворков с детальной информацией
        """
        if not text:
            return []

        # Если нет LLM, используем fallback метод
        if self.llm is None:
            return self.extract_frameworks_fallback(text, paper_url)

        try:
            # Создаем парсер с Pydantic моделью
            pydantic_parser = PydanticOutputParser(pydantic_object=FrameworksList)
            
            # Оборачиваем в OutputFixingParser для автоматического исправления ошибок, если доступен
            if OutputFixingParser is not None:
                fixing_parser = OutputFixingParser.from_llm(
                    parser=pydantic_parser,
                    llm=self.llm
                )
            else:
                fixing_parser = None
            
            prompt_template = """Ты — помощник, который извлекает детальную информацию об AutoML фреймворках и библиотеках из научных текстов.

Твоя задача — найти все упоминания **конкретных систем/библиотек/фреймворков AutoML** в тексте и для каждого извлечь следующую информацию:

1. **Название** фреймворка (точное название из текста)
2. **Описание** — назначение, основные возможности, что делает фреймворк
3. **Преимущества** — чем фреймворк лучше других, его уникальные особенности
4. **Сильные стороны** — в чем фреймворк преуспевает
5. **Слабые стороны** — ограничения, недостатки, что не умеет делать
6. **Типы данных** — с какими типами данных работает (tabular, image, text, multimodal, time series и т.д.)
7. **Что не поддерживает** — явные ограничения, неподдерживаемые функции
8. **Результаты в бенчмарках** — места в рейтингах, метрики производительности, типы датасетов, на которых тестировался
9. **Код из статьи** — примеры кода, API, фрагменты кода, связанные с фреймворком
10. **Ссылки на репозитории** — GitHub, GitLab и другие ссылки на исходный код

Правила:
- Используй **только информацию из текста** — не придумывай данные, которых там нет
- Если информация отсутствует, оставь поле пустым или используй пустой список
- Для каждого фреймворка создай отдельную запись
- Извлекай код точно как он написан в тексте
- Для ссылок на репозитории ищи упоминания GitHub, GitLab, URLs в тексте
- Результаты бенчмарков извлекай из таблиц, графиков и текстовых описаний

Текст статьи:
{text}

{format_instructions}

Если подходящих фреймворков нет, верни пустой список в поле frameworks."""

            format_instructions = pydantic_parser.get_format_instructions()
            prompt = prompt_template.format(text=text, format_instructions=format_instructions)

            message = HumanMessage(content=prompt)
            response = await self.llm.ainvoke([message])
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Отладочный вывод (можно убрать в продакшене)
            print(f"[DEBUG] LLM response for framework extraction: {response_text[:500]}...")
            
            # Используем OutputFixingParser для парсинга с автоматическим исправлением ошибок, если доступен
            if fixing_parser is not None:
                try:
                    parsed_result = fixing_parser.parse(response_text)
                    if isinstance(parsed_result, FrameworksList):
                        # Добавляем URL статьи к каждому фреймворку
                        for framework in parsed_result.frameworks:
                            if not framework.paper_url:
                                framework.paper_url = paper_url
                        result = parsed_result.frameworks
                        print(f"[DEBUG] Extracted {len(result)} frameworks with detailed info")
                        return result
                except Exception as parse_error:
                    print(f"[DEBUG] OutputFixingParser error: {parse_error}, trying direct parsing")
            
            # Если OutputFixingParser недоступен или произошла ошибка, используем прямой парсинг с повторными попытками
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    parsed_result = pydantic_parser.parse(response_text)
                    if isinstance(parsed_result, FrameworksList):
                        # Добавляем URL статьи к каждому фреймворку
                        for framework in parsed_result.frameworks:
                            if not framework.paper_url:
                                framework.paper_url = paper_url
                        result = parsed_result.frameworks
                        print(f"[DEBUG] Extracted {len(result)} frameworks (Pydantic parser, attempt {attempt + 1})")
                        return result
                except Exception as parse_error:
                    if attempt < max_retries - 1:
                        # Пытаемся исправить ошибку, запросив LLM исправить ответ
                        print(f"[DEBUG] Parser error (attempt {attempt + 1}): {parse_error}")
                        fix_prompt = f"""Предыдущий ответ содержал ошибку при парсинге. Исправь его, чтобы он соответствовал требуемому формату.

Ошибка парсинга: {str(parse_error)}

Предыдущий ответ:
{response_text}

Требуемый формат:
{format_instructions}

Исправленный ответ (только валидный JSON, без дополнительного текста):"""
                        fix_message = HumanMessage(content=fix_prompt)
                        fix_response = await self.llm.ainvoke([fix_message])
                        response_text = fix_response.content if hasattr(fix_response, "content") else str(fix_response)
                        print(f"[DEBUG] Fixed response: {response_text[:200]}...")
                    else:
                        # Последняя попытка не удалась, используем fallback
                        print(f"[DEBUG] Parser error after {max_retries} attempts: {parse_error}, using fallback")
                        return self.extract_frameworks_fallback(text, paper_url)
            
            # Если все попытки не удались
            return self.extract_frameworks_fallback(text, paper_url)
            
        except AuthenticationError as e:
            error_msg = (
                f"Ошибка аутентификации API: {str(e)}\n"
                "Проверьте правильность API ключа в переменных окружения."
            )
            print(error_msg)
            return []
        except (ValueError, OSError, Exception) as e:
            print(f"Ошибка при извлечении фреймворков: {e}")
            return []

    async def extract_frameworks_from_paper(self, paper_info: PaperInfo, extract_from_abstract: bool = True) -> dict[str, Any]:
        """Извлечение информации о фреймворках из статьи.
        
        Args:
            paper_info: Информация о статье
            extract_from_abstract:  Если True, извлекает текст из абстракта.
                                    Если False, извлекает текст из полного PDF.
        
        Returns:
            Словарь с информацией о статье и извлеченными фреймворками
        """
        if extract_from_abstract:
            # Используем абстракт
            text_for_extraction = paper_info.abstract
            print(f"[DEBUG] Extracting frameworks from abstract (length: {len(text_for_extraction)} chars)")
        else:
            # Извлекаем текст из PDF
            if paper_info.pdf_url:
                text_for_extraction = await self.extract_text_from_pdf(paper_info.pdf_url)
                print(f"[DEBUG] Extracting frameworks from PDF (length: {len(text_for_extraction)} chars)")
                
                # Если PDF не удалось извлечь (пустой текст), используем абстракт как fallback
                if not text_for_extraction or not text_for_extraction.strip():
                    print("[WARNING] PDF extraction failed or returned empty text, falling back to abstract")
                    text_for_extraction = paper_info.abstract
                    print(f"[DEBUG] Using abstract instead (length: {len(text_for_extraction)} chars)")
            else:
                # Если PDF URL отсутствует, используем абстракт как fallback
                text_for_extraction = paper_info.abstract
                print(f"[DEBUG] PDF URL not available, using abstract instead (length: {len(text_for_extraction)} chars)")
        
        # Извлекаем детальную информацию о фреймворках из выбранного текста
        all_frameworks = await self.extract_frameworks(text_for_extraction, paper_url=paper_info.url)
        print(f"[DEBUG] Extracted {len(all_frameworks)} frameworks with detailed info")

        return {
            "title": paper_info.title,
            "authors": paper_info.authors,
            "abstract": paper_info.abstract,
            "url": paper_info.url,
            "source": paper_info.source,
            "published_date": paper_info.published_date,
            "frameworks": [fw.model_dump() if hasattr(fw, 'model_dump') else fw for fw in all_frameworks],
        }


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
            self._summarizer = PaperSummarizer(llm=llm)
        else:
            self._summarizer = None
        self._graph = self._build_graph()

    @staticmethod
    def _create_llm(llm_config: LLMConfig | None = None) -> ChatOpenAI:
        """Создает экземпляр ChatOpenAI из конфигурации.
        
        Всегда использует OpenRouter из .env файла. Модель берется из llm_config или
        из settings.agents для этого агента.
        
        Args:
            llm_config: Конфигурация LLM. Если не указана, используется из settings.agents
                        для этого агента или дефолтная из .env.
        
        Returns:
            Экземпляр ChatOpenAI
        
        Raises:
            ValueError: Если API ключ не найден.
        """
        # Используем переданную конфигурацию или конфигурацию из settings.agents
        if llm_config is None:
            # Пытаемся получить конфигурацию для этого агента из settings.agents
            agent_config = getattr(settings.agents, "paper_researcher", None)
            if agent_config:
                llm_config = agent_config
            else:
                # Загружаем из переменных окружения (всегда OpenRouter)
                llm_config = LLMConfig.from_env()

        # Проверяем API ключ
        api_key = (
            llm_config.api_key
            or os.getenv("MAS_LLM__API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "API key не найден. Установите его в файле .env:\n"
                "OPENROUTER_API_KEY=your-openrouter-api-key"
            )

        # Убеждаемся, что base_url установлен для OpenRouter
        base_url = (
            llm_config.base_url
            or os.getenv("BASE_URL")
            or os.getenv("MAS_LLM__BASE_URL")
            or "https://openrouter.ai/api/v1"
        )

        return ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,
            base_url=base_url,
        )

    def _build_graph(self):
        """Построение графа состояний для поиска и извлечения информации о фреймворках."""
        workflow = StateGraph[PaperSearchState, None, PaperSearchState, PaperSearchState](PaperSearchState)

        # Добавляем узлы
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("extract_frameworks", self._extract_frameworks_node)
        workflow.add_node("merge_frameworks", self._merge_frameworks_node)

        # Определяем граф переходов
        workflow.set_entry_point("search_papers")
        workflow.add_edge("search_papers", "extract_frameworks")
        workflow.add_edge("extract_frameworks", "merge_frameworks")
        workflow.add_edge("merge_frameworks", END)

        return workflow.compile()

    async def _search_papers_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для поиска статей."""
        query = state["query"]
        # Используем max_results из состояния, если указан, иначе используем дефолтные значения
        max_results = state.get("max_results")
        if max_results is None:
            # В тестовом режиме ищем меньше статей
            max_results = 2 if self._test_mode else 6
        sort_by = state.get("sort_by")
        days_back = state.get("days_back")
        
        papers_arxiv = await self._search_tools.search_arxiv(
            query,
            max_results=max_results,
            sort_by=sort_by,
            days_back=days_back,
        )

        all_papers = papers_arxiv
        papers_dict = [
            {
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "url": p.url,
                "source": p.source,
                "pdf_url": p.pdf_url,
                "published_date": p.published_date,
            }
            for p in all_papers
        ]

        return {
            **state,
            "papers": papers_dict,
            "current_paper_index": 0,
            "paper_frameworks": [],
            "frameworks": [],
        }

    async def _extract_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для извлечения информации о фреймворках из статей."""
        papers = state["papers"]
        paper_frameworks_list = []

        for paper_dict in papers:
            paper_info = PaperInfo(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                abstract=paper_dict["abstract"],
                url=paper_dict["url"],
                source=paper_dict["source"],
                pdf_url=paper_dict.get("pdf_url"),
                published_date=paper_dict.get("published_date"),
            )
            # В обычном режиме извлекаем из полного текста PDF, а не из абстракта
            if not self._test_mode:
                paper_data = await self._summarizer.extract_frameworks_from_paper(paper_info, extract_from_abstract=False)
            else:
                paper_data = await self._summarizer.extract_frameworks_from_paper(paper_info, extract_from_abstract=True)
            paper_frameworks_list.append(paper_data)
            
        return {
            **state,
            "paper_frameworks": paper_frameworks_list,
        }

    async def _merge_frameworks_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для объединения информации о фреймворках из разных статей.
        
        Объединяет информацию о фреймворках с одинаковыми названиями из разных статей.
        """
        paper_frameworks_list = state.get("paper_frameworks", [])
        frameworks_dict: dict[str, dict[str, Any]] = {}
        
        # Собираем все фреймворки и объединяем информацию о фреймворках с одинаковыми названиями
        for paper_data in paper_frameworks_list:
            frameworks = paper_data.get("frameworks", [])
            paper_url = paper_data.get("url", "")
            
            for framework_data in frameworks:
                # framework_data может быть dict или FrameworkInfo
                if isinstance(framework_data, dict):
                    framework_name = framework_data.get("name", "")
                else:
                    framework_name = getattr(framework_data, "name", "")
                
                if not framework_name:
                    continue
                
                # Если фреймворк уже встречался, объединяем информацию
                if framework_name in frameworks_dict:
                    existing = frameworks_dict[framework_name]
                    # Объединяем списки (advantages, strengths, weaknesses и т.д.)
                    for key in ["advantages", "strengths", "weaknesses", "supported_data_types", 
                               "unsupported_features", "benchmark_results", "code_snippets", "repository_urls"]:
                        existing_list = existing.get(key, [])
                        new_list = framework_data.get(key, []) if isinstance(framework_data, dict) else getattr(framework_data, key, [])
                        # Объединяем списки, убирая дубликаты
                        combined = list(set(existing_list + new_list))
                        existing[key] = combined
                    
                    # Обновляем описание, если новое более полное
                    new_description = framework_data.get("description", "") if isinstance(framework_data, dict) else getattr(framework_data, "description", "")
                    if new_description and len(new_description) > len(existing.get("description", "")):
                        existing["description"] = new_description
                    
                    # Добавляем URL статьи, если его еще нет
                    paper_urls = existing.get("paper_urls", [])
                    # Также проверяем paper_url (единственное число) для обратной совместимости
                    existing_paper_url = existing.get("paper_url", "")
                    if existing_paper_url and existing_paper_url not in paper_urls:
                        paper_urls.append(existing_paper_url)
                    if paper_url and paper_url not in paper_urls:
                        paper_urls.append(paper_url)
                    existing["paper_urls"] = paper_urls
                else:
                    # Первое упоминание фреймворка
                    if isinstance(framework_data, dict):
                        framework_dict = framework_data.copy()
                    else:
                        framework_dict = framework_data.model_dump() if hasattr(framework_data, 'model_dump') else {}
                    
                    # Добавляем список URL статей
                    # Сохраняем и paper_url (единственное) для обратной совместимости, и paper_urls (множественное)
                    if paper_url:
                        framework_dict["paper_urls"] = [paper_url]
                        # Если paper_url уже есть в словаре, не перезаписываем его
                        if "paper_url" not in framework_dict:
                            framework_dict["paper_url"] = paper_url
                    else:
                        framework_dict["paper_urls"] = []
                    frameworks_dict[framework_name] = framework_dict
        
        # Преобразуем словарь в список
        frameworks_list = list(frameworks_dict.values())

        return {
            **state,
            "frameworks": frameworks_list,
        }

    async def handle(
        self,
        query: str,
        sort_by: arxiv.SortCriterion | None = None,
        days_back: int | None = None,
        max_results: int | None = None,
    ) -> PaperSearchState:
        """Обработка запроса на поиск и извлечение информации о фреймворках.
        
        Args:
            query: Строка с запросом для поиска статей.
            sort_by: Критерий сортировки результатов:
                     - arxiv.SortCriterion.Relevance (по умолчанию) - по релевантности
                     - arxiv.SortCriterion.SubmittedDate - по дате подачи (новые сначала)
                     - arxiv.SortCriterion.LastUpdatedDate - по дате обновления
            days_back: Если указано, возвращает только статьи, опубликованные за последние N дней.
                      Полезно для поиска новых статей. Рекомендуется использовать вместе с
                      sort_by=arxiv.SortCriterion.SubmittedDate.
            max_results: Максимальное количество статей для поиска. По умолчанию: 2 в тестовом режиме,
                        6 в обычном режиме. Можно увеличить для более полного анализа (например, 50-100).
        
        Returns:
            PaperSearchState с результатами поиска и структурированной информацией о фреймворках.
            Основная информация находится в поле "frameworks" - список словарей с детальной информацией
            о каждом найденном фреймворке.
        """
        if not query:
            return {
                "query": "",
                "papers": [],
                "paper_frameworks": [],
                "current_paper_index": 0,
                "error": "Не указан запрос для поиска статей.",
                "sort_by": None,
                "days_back": None,
                "max_results": None,
                "frameworks": [],
            }

        try:
            # Инициализируем начальное состояние
            initial_state: PaperSearchState = {
                "query": query,
                "papers": [],
                "paper_frameworks": [],
                "current_paper_index": 0,
                "error": None,
                "sort_by": sort_by,
                "days_back": days_back,
                "max_results": max_results,
                "frameworks": [],
            }

            # Запускаем граф
            final_state = await self._graph.ainvoke(initial_state)
            return final_state
        except (ValueError, OSError, KeyError) as e:
            return {
                "query": query,
                "papers": [],
                "paper_frameworks": [],
                "current_paper_index": 0,
                "error": f"Ошибка при поиске статей: {str(e)}",
                "sort_by": sort_by,
                "days_back": days_back,
                "max_results": max_results,
                "frameworks": [],
            }

