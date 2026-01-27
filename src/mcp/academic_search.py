"""Academic Search MCP Server."""
import asyncio
import json
import logging
from typing import Any, Dict, List

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
import arxiv
from semanticscholar import SemanticScholar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AcademicSearchServer:
    """MCP Server for academic paper search."""

    def __init__(self):
        self.server = Server("academic-search")
        self.sch = SemanticScholar()
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="search_arxiv",
                    description="Search for papers on arXiv.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 5},
                            "sort_by": {"type": "string", "enum": ["relevance", "lastUpdatedDate", "submittedDate"], "default": "relevance"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="search_semantic_scholar",
                    description="Search for papers on Semantic Scholar.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            if name == "search_arxiv":
                return await self._search_arxiv(arguments)
            elif name == "search_semantic_scholar":
                return await self._search_semantic_scholar(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_arxiv(self, args: Dict[str, Any]) -> list[types.TextContent]:
        query = args["query"]
        max_results = args.get("max_results", 5)
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        sort_by = sort_map.get(args.get("sort_by", "relevance"), arxiv.SortCriterion.Relevance)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        # Run in thread to avoid blocking loop with network calls
        def run_search():
            results = []
            for r in client.results(search):
                results.append({
                    "title": r.title,
                    "authors": [a.name for a in r.authors],
                    "year": r.published.year,
                    "pdf_url": r.pdf_url,
                    "summary": r.summary,
                    "id": r.entry_id
                })
            return results

        results = await asyncio.to_thread(run_search)
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    async def _search_semantic_scholar(self, args: Dict[str, Any]) -> list[types.TextContent]:
        query = args["query"]
        limit = args.get("limit", 5)
        
        def run_search():
            return self.sch.search_paper(query, limit=limit)

        results_obj = await asyncio.to_thread(run_search)
        
        results = []
        for r in results_obj:
            results.append({
                "title": r.title,
                "authors": [a.name for a in r.authors] if r.authors else [],
                "year": r.year,
                "url": r.url,
                "abstract": r.abstract,
                "paperId": r.paperId,
                "venue": r.venue
            })

        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    async def run(self):
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Academic Search MCP server starting...")
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

def main():
    server = AcademicSearchServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
