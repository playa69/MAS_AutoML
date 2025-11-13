"""Агент для поиска и саммаризации научных статей с использованием LangGraph и LangChain."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict
from urllib.error import URLError

import arxiv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from openai import AuthenticationError

from ...config.settings import LLMConfig, settings


class PaperSearchState(TypedDict):
    """Состояние для графа поиска и саммаризации статей."""

    query: str
    papers: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    current_paper_index: int
    error: str | None
    sort_by: arxiv.SortCriterion | None
    days_back: int | None  # Фильтр по дате: только статьи за последние N дней


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

    def __init__(self, llm: ChatOpenAI) -> None:
        """Инициализация саммаризатора.
        
        Args:
            llm: Готовый экземпляр ChatOpenAI для саммаризации.
        """
        self.llm = llm

    async def summarize_abstract(self, abstract: str) -> str:
        """Саммаризация абстракта статьи."""
        if not abstract:
            return "Абстракт отсутствует."

        try:
            prompt = f"""Создай краткое резюме следующего абстракта научной статьи на русском языке.
            Выдели основные идеи, методы и выводы.

            Абстракт:
            {abstract}

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
            print(f"Ошибка при саммаризации абстракта: {e}")
            return f"Ошибка саммаризации: {str(e)}"

    async def summarize_paper(self, paper_info: PaperInfo) -> dict[str, Any]:
        """Саммаризация полной информации о статье."""
        summary = await self.summarize_abstract(paper_info.abstract)

        return {
            "title": paper_info.title,
            "authors": paper_info.authors,
            "abstract": paper_info.abstract,
            "summary": summary,
            "url": paper_info.url,
            "source": paper_info.source,
            "published_date": paper_info.published_date,
        }


class PaperResearchAgent:
    """Агент для поиска и саммаризации научных статей с использованием LangGraph."""

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
                        - Пропускает саммаризацию, возвращает только абстракты
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
        """Построение графа состояний для поиска и саммаризации."""
        workflow = StateGraph[PaperSearchState, None, PaperSearchState, PaperSearchState](PaperSearchState)

        # Добавляем узлы
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("summarize_papers", self._summarize_papers_node)

        # Определяем граф переходов
        workflow.set_entry_point("search_papers")
        workflow.add_edge("search_papers", "summarize_papers")
        workflow.add_edge("summarize_papers", END)

        return workflow.compile()

    async def _search_papers_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для поиска статей."""
        query = state["query"]
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
        }

    async def _summarize_papers_node(self, state: PaperSearchState) -> PaperSearchState:
        """Узел для саммаризации статей."""
        papers = state["papers"]
        summaries = []

        # В тестовом режиме пропускаем саммаризацию, возвращаем только абстракты
        if self._test_mode:
            for paper_dict in papers:
                summaries.append({
                    "title": paper_dict["title"],
                    "authors": paper_dict["authors"],
                    "abstract": paper_dict["abstract"],
                    "summary": paper_dict["abstract"],  # В тестовом режиме summary = abstract
                    "url": paper_dict["url"],
                    "source": paper_dict["source"],
                    "published_date": paper_dict.get("published_date"),
                })
        else:
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
                summary = await self._summarizer.summarize_paper(paper_info)
                summaries.append(summary)

        return {
            **state,
            "summaries": summaries,
        }

    async def handle(
        self,
        query: str,
        sort_by: arxiv.SortCriterion | None = None,
        days_back: int | None = None,
    ) -> PaperSearchState:
        """Обработка запроса на поиск и саммаризацию статей.
        
        Args:
            query: Строка с запросом для поиска статей.
            sort_by: Критерий сортировки результатов:
                     - arxiv.SortCriterion.Relevance (по умолчанию) - по релевантности
                     - arxiv.SortCriterion.SubmittedDate - по дате подачи (новые сначала)
                     - arxiv.SortCriterion.LastUpdatedDate - по дате обновления
            days_back: Если указано, возвращает только статьи, опубликованные за последние N дней.
                      Полезно для поиска новых статей. Рекомендуется использовать вместе с
                      sort_by=arxiv.SortCriterion.SubmittedDate.
        
        Returns:
            PaperSearchState с результатами поиска и саммаризации статей.
        """
        if not query:
            return {
                "query": "",
                "papers": [],
                "summaries": [],
                "current_paper_index": 0,
                "error": "Не указан запрос для поиска статей.",
                "sort_by": None,
                "days_back": None,
            }

        try:
            # Инициализируем начальное состояние
            initial_state: PaperSearchState = {
                "query": query,
                "papers": [],
                "summaries": [],
                "current_paper_index": 0,
                "error": None,
                "sort_by": sort_by,
                "days_back": days_back,
            }

            # Запускаем граф
            final_state = await self._graph.ainvoke(initial_state)
            return final_state
        except (ValueError, OSError, KeyError) as e:
            return {
                "query": query,
                "papers": [],
                "summaries": [],
                "current_paper_index": 0,
                "error": f"Ошибка при поиске статей: {str(e)}",
                "sort_by": sort_by,
                "days_back": days_back,
            }

