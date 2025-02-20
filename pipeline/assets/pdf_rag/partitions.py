import dagster as dg

subdirectory_partitions = dg.DynamicPartitionsDefinition(name="subdirectories")
