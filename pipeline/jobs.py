import dagster as dg

pdf_rag_job = dg.define_asset_job(
    "pdf_rag_job",
    selection=dg.AssetSelection.groups("pdf_rag"),
)
