from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import time
from typing import Any, Dict

from .core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
)

app = FastAPI(title="EDA Quality API", version="0.1.0")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/quality")
async def quality_from_json(data: Dict[str, Any]):
    """
    Пример: {"n_rows": 1000, "max_missing_share": 0.05, "has_constant_columns": false}
    """
    start = time.time()
    try:
        n_rows = data.get("n_rows", 0)
        max_missing_share = data.get("max_missing_share", 0.0)
        has_constant_columns = data.get("has_constant_columns", False)

        # Пример простой логики
        ok_for_model = (
            n_rows >= 100
            and max_missing_share <= 0.1
            and not has_constant_columns
        )
        quality_score = 1.0 - max_missing_share
        if n_rows < 100:
            quality_score -= 0.2
        if has_constant_columns:
            quality_score -= 0.1
        quality_score = max(0.0, min(1.0, quality_score))

        latency_ms = int((time.time() - start) * 1000)

        return {
            "ok_for_model": ok_for_model,
            "quality_score": round(quality_score, 3),
            "latency_ms": latency_ms,
            "flags": {
                "too_few_rows": n_rows < 100,
                "too_many_missing": max_missing_share > 0.5,
                "has_constant_columns": has_constant_columns,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")


@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)):
    """
    Аналогичен /quality, но читает CSV и применяет ваш EDA-стек.
    """
    start = time.time()
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        ok_for_model = (
            not flags["too_few_rows"]
            and not flags["too_many_missing"]
            and not flags.get("has_constant_columns", False)
            and not flags.get("has_high_cardinality_categoricals", False)
        )

        latency_ms = int((time.time() - start) * 1000)

        return {
            "ok_for_model": ok_for_model,
            "quality_score": round(flags["quality_score"], 3),
            "latency_ms": latency_ms,
            "flags": flags,
        }
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")


# === НОВЫЙ ЭНДПОИНТ ИЗ HW04 (обязательный) ===
@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    Возвращает ТОЛЬКО флаги качества, включая ваши новые эвристики.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        # Убираем всё лишнее, оставляем только флаги
        return {"flags": flags}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")