from dagster import Definitions, load_assets_from_package_module

from pipeline import assets

defs = Definitions(
    assets=load_assets_from_package_module(assets),
)
