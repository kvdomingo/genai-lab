from dagster import Definitions, load_assets_from_package_module

from pipeline.assets import basic_qa_rag, obsidian_qa_rag, pdf_rag
from pipeline.jobs import pdf_rag_job
from pipeline.resources import RESOURCES
from pipeline.sensors import generate_dynamic_subdirectory_partitions_sensor

defs = Definitions(
    assets=[
        *load_assets_from_package_module(basic_qa_rag, "basic_qa_rag", "basic_qa_rag"),
        *load_assets_from_package_module(
            obsidian_qa_rag, "obsidian_qa_rag", "obsidian_qa_rag"
        ),
        *load_assets_from_package_module(pdf_rag, "pdf_rag", "pdf_rag"),
    ],
    jobs=[pdf_rag_job],
    sensors=[generate_dynamic_subdirectory_partitions_sensor],
    resources=RESOURCES,
)
