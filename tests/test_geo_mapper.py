"""Tests for geographic mapping dictionaries."""

from salaries_analysis.data.geo_mapper import PROVINCE_MAPPER, REGIONS_MAPPER

EXPECTED_REGIONS = 20
EXPECTED_PROVINCES = 107
EXPECTED_MACRO_REGIONS = {"Nord", "Centro", "Sud"}


class TestRegionsMapper:
    """Tests for the REGIONS_MAPPER dictionary."""

    def test_all_20_regions_present(self) -> None:
        assert len(REGIONS_MAPPER) == EXPECTED_REGIONS

    def test_all_macro_regions_used(self) -> None:
        macro_regions = set(REGIONS_MAPPER.values())
        assert macro_regions == EXPECTED_MACRO_REGIONS

    def test_lombardia_is_nord(self) -> None:
        assert REGIONS_MAPPER["Lombardia"] == "Nord"

    def test_lazio_is_centro(self) -> None:
        assert REGIONS_MAPPER["Lazio"] == "Centro"

    def test_sicilia_is_sud(self) -> None:
        assert REGIONS_MAPPER["Sicilia"] == "Sud"


class TestProvinceMapper:
    """Tests for the PROVINCE_MAPPER dictionary."""

    def test_all_provinces_map_to_valid_regions(self) -> None:
        for province, region in PROVINCE_MAPPER.items():
            assert region in REGIONS_MAPPER, (
                f"Province '{province}' maps to unknown region '{region}'"
            )

    def test_all_107_provinces_present(self) -> None:
        assert len(PROVINCE_MAPPER) == EXPECTED_PROVINCES

    def test_every_region_has_a_province(self) -> None:
        assert set(PROVINCE_MAPPER.values()) == set(REGIONS_MAPPER)

    def test_major_cities_present(self) -> None:
        expected = ["Milano", "Roma", "Napoli", "Torino", "Firenze", "Bologna", "Bari"]
        for city in expected:
            assert city in PROVINCE_MAPPER, f"Missing major city: {city}"

    def test_milano_maps_to_lombardia(self) -> None:
        assert PROVINCE_MAPPER["Milano"] == "Lombardia"

    def test_roma_maps_to_lazio(self) -> None:
        assert PROVINCE_MAPPER["Roma"] == "Lazio"

    def test_no_duplicate_provinces(self) -> None:
        provinces = list(PROVINCE_MAPPER.keys())
        assert len(provinces) == len(set(provinces))
