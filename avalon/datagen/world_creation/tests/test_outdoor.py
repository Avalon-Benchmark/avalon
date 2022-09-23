from pathlib import Path

import attr

from avalon.contrib.testing_utils import create_temp_file_path
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import temp_path_
from avalon.contrib.testing_utils import use
from avalon.datagen.world_creation.configs.export import get_eval_agent_export_config
from avalon.datagen.world_creation.tests.fixtures import ChecksumManifest
from avalon.datagen.world_creation.tests.fixtures import outdoor_world_catalog_id_
from avalon.datagen.world_creation.tests.fixtures import outdoor_worlds_manifest_
from avalon.datagen.world_creation.tests.generate_references import get_path_checksum
from avalon.datagen.world_creation.tests.helpers import compare_files
from avalon.datagen.world_creation.tests.helpers import create_world
from avalon.datagen.world_creation.tests.helpers import get_reference_data_path
from avalon.datagen.world_creation.tests.helpers import make_world_file_generic
from avalon.datagen.world_creation.tests.params import OUTDOOR_WORLD_CATALOG
from avalon.datagen.world_creation.tests.params import OutdoorWorldParams


@slow_integration_test
@use(outdoor_world_catalog_id_)
def test_world_is_reproducible(outdoor_world_catalog_id: str):
    outdoor_world_params: OutdoorWorldParams = OUTDOOR_WORLD_CATALOG[outdoor_world_catalog_id]
    export_config = attr.evolve(get_eval_agent_export_config(), is_tiled=False)

    with create_temp_file_path() as export_path_a, create_temp_file_path() as export_path_b:
        export_path_a.mkdir(parents=True)
        export_path_b.mkdir(parents=True)
        create_world(*outdoor_world_params, export_path=export_path_a, export_config=export_config)
        create_world(*outdoor_world_params, export_path=export_path_b, export_config=export_config)
        # World files can contain absolute paths that cause spurious diffs, so we replace them with
        for file in export_path_a.iterdir():
            path_a = file
            path_b = export_path_b / file.name
            make_world_file_generic(path_a)
            make_world_file_generic(path_b)
            compare_files(path_a, path_b)


@slow_integration_test
@use(temp_path_, outdoor_world_catalog_id_, outdoor_worlds_manifest_)
def test_outdoor_world_matches_reference(
    temp_path: Path, outdoor_world_catalog_id: str, outdoor_worlds_manifest: ChecksumManifest
):
    outdoor_world_params: OutdoorWorldParams = OUTDOOR_WORLD_CATALOG[outdoor_world_catalog_id]
    create_world(*outdoor_world_params, export_path=temp_path)
    for path in temp_path.iterdir():
        make_world_file_generic(path)
    generated_world_checksum = get_path_checksum(temp_path)
    reference_world_checksum = outdoor_worlds_manifest["checksums"][outdoor_world_catalog_id]
    if generated_world_checksum != reference_world_checksum:
        all_reference_world_path = get_reference_data_path(
            "outdoor_worlds", outdoor_worlds_manifest["snapshot_commit"]
        )
        reference_world_path = all_reference_world_path / outdoor_world_catalog_id
        num_generated_files = len(list(temp_path.iterdir()))
        num_reference_files = len(list(reference_world_path.iterdir()))
        assert num_generated_files == num_reference_files, "Generated and reference file count must be the same"
        for reference_file_path in reference_world_path.iterdir():
            generated_file_path = temp_path / reference_file_path.relative_to(reference_world_path)
            compare_files(generated_file_path, reference_file_path)
    else:
        assert generated_world_checksum == reference_world_checksum, "Generated world checksum does not match manifest"
