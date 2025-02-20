import dagster as dg

from common.settings import settings
from pipeline.assets.pdf_rag.partitions import subdirectory_partitions


@dg.sensor(minimum_interval_seconds=settings.SENSOR_MIN_INTERVAL_SECONDS)
def generate_dynamic_subdirectory_partitions_sensor():
    subdirs = [d.name for d in settings.RAG_ROOT_DIR.iterdir() if d.is_dir()]

    return dg.SensorResult(
        dynamic_partitions_requests=[subdirectory_partitions.build_add_request(subdirs)]
    )
