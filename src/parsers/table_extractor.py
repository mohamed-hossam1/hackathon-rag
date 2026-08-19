import logging
from typing import Any, List, Optional

from src.models.document import TableData

logger = logging.getLogger("medical_rag.parsers.table_extractor")


class TableExtractor:
    """Extractor for detecting and converting document tables into normalized Markdown format."""

    @staticmethod
    def to_markdown_table(headers: Optional[List[str]], rows: List[List[str]]) -> str:
        """Converts header and row lists into pipe-delimited Markdown table text.

        Args:
            headers: Optional list of column header titles.
            rows: 2D list of row cell values.

        Returns:
            Formatted pipe-delimited Markdown string.
        """
        if not rows and not headers:
            return ""

        lines = []
        col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
        if col_count == 0:
            return ""

        # Construct Header Row
        if headers:
            clean_headers = [str(h).strip().replace("\n", " ").replace("|", "\\|") for h in headers]
            # Fill missing column header names if rows have more columns
            while len(clean_headers) < col_count:
                clean_headers.append(f"Header_{len(clean_headers)+1}")
            lines.append("| " + " | ".join(clean_headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")
        else:
            dummy_headers = [f"Col_{i+1}" for i in range(col_count)]
            lines.append("| " + " | ".join(dummy_headers) + " |")
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

        # Construct Data Rows
        for row in rows:
            clean_cells = [str(c).strip().replace("\n", " ").replace("|", "\\|") if c is not None else "" for c in row]
            # Pad short rows
            while len(clean_cells) < col_count:
                clean_cells.append("")
            lines.append("| " + " | ".join(clean_cells[:col_count]) + " |")

        return "\n".join(lines)

    def extract_from_pymupdf_page(self, page_obj: Any) -> List[TableData]:
        """Extracts tables from a PyMuPDF (fitz) page object.

        Args:
            page_obj: fitz.Page object.

        Returns:
            List of TableData domain models.
        """
        tables_data: List[TableData] = []
        try:
            # Call PyMuPDF built-in find_tables()
            finder = page_obj.find_tables()
            if not finder or not finder.tables:
                return []

            for idx, table in enumerate(finder.tables):
                extracted_rows = table.extract()
                if not extracted_rows:
                    continue

                headers: Optional[List[str]] = None
                data_rows: List[List[str]] = []

                # Header detection heuristic (first row)
                if len(extracted_rows) > 1:
                    headers = [str(cell) if cell is not None else "" for cell in extracted_rows[0]]
                    data_rows = [[str(cell) if cell is not None else "" for cell in row] for row in extracted_rows[1:]]
                else:
                    data_rows = [[str(cell) if cell is not None else "" for cell in row] for row in extracted_rows]

                normalized_md = self.to_markdown_table(headers, data_rows)

                tables_data.append(
                    TableData(
                        table_index=idx,
                        headers=headers,
                        rows=data_rows,
                        caption=None,
                        normalized_text=normalized_md
                    )
                )
            logger.info(f"Extracted {len(tables_data)} tables from PyMuPDF page")
        except Exception as err:
            logger.warning(f"Error extracting tables from PyMuPDF page: {err}")

        return tables_data

    def extract_from_docx_tables(self, docx_tables: List[Any]) -> List[TableData]:
        """Extracts tables from python-docx Table objects.

        Args:
            docx_tables: List of docx.table.Table objects.

        Returns:
            List of TableData domain models.
        """
        tables_data: List[TableData] = []
        for idx, table in enumerate(docx_tables):
            try:
                all_rows = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    all_rows.append(row_cells)

                if not all_rows:
                    continue

                headers = all_rows[0] if len(all_rows) > 1 else None
                data_rows = all_rows[1:] if len(all_rows) > 1 else all_rows

                normalized_md = self.to_markdown_table(headers, data_rows)

                tables_data.append(
                    TableData(
                        table_index=idx,
                        headers=headers,
                        rows=data_rows,
                        caption=None,
                        normalized_text=normalized_md
                    )
                )
            except Exception as err:
                logger.warning(f"Error extracting docx table #{idx}: {err}")

        return tables_data
