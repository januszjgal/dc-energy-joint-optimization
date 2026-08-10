from __future__ import annotations

import io
import unittest
import zipfile

import pandas as pd

from energy_model_v3.adapters.parsers import (
    compare_ercot_renewables,
    parse_caiso_da,
    parse_ercot_da,
    parse_ercot_sced_executions,
    parse_miso_da,
    parse_nyiso_da,
    parse_pjm_da,
    parse_spp_da,
    read_ercot_native_load_xlsx,
    select_ercot_document,
    select_ercot_sced_document,
    time_weight_ercot_sced_hourly,
)
from energy_model_v3.cli import _ercot_native_load_years


def native_load_workbook(rows: list[list[str]]) -> bytes:
    headers = [
        "Hour Ending",
        "COAST",
        "EAST",
        "FWEST",
        "NORTH",
        "NCENT",
        "SOUTH",
        "SCENT",
        "WEST",
        "ERCOT",
    ]
    shared = headers + [row[0] for row in rows]
    strings = "".join(f"<si><t>{value}</t></si>" for value in shared)
    sheet_rows = [
        "<row r=\"1\">"
        + "".join(
            f'<c r="{chr(65 + index)}1" t="s"><v>{index}</v></c>'
            for index in range(len(headers))
        )
        + "</row>"
    ]
    for row_number, row in enumerate(rows, start=2):
        cells = [
            f'<c r="A{row_number}" t="s">'
            f"<v>{len(headers) + row_number - 2}</v></c>"
        ]
        cells.extend(
            f'<c r="{chr(65 + index)}{row_number}"><v>{value}</v></c>'
            for index, value in enumerate(row[1:], start=1)
        )
        sheet_rows.append(
            f'<row r="{row_number}">' + "".join(cells) + "</row>"
        )
    namespace = (
        "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    )
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                f'<sst xmlns="{namespace}">{strings}</sst>',
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                f'<worksheet xmlns="{namespace}"><sheetData>'
                + "".join(sheet_rows)
                + "</sheetData></worksheet>",
            )
        return buffer.getvalue()


class LockedSourceParserTests(unittest.TestCase):
    def test_ercot_load_archives_follow_central_year_boundaries(self) -> None:
        years = _ercot_native_load_years(
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-02T00:00:00Z"),
        )
        self.assertEqual(years, [2025, 2026])

    def test_pjm_dom_parser(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime_beginning_utc": ["2025-09-01T00:00:00Z"],
                "pnode_id": [34964545],
                "total_lmp_da": [42.5],
                "system_energy_price_da": [40.0],
                "congestion_price_da": [2.0],
                "marginal_loss_price_da": [0.5],
                "is_verified": [True],
                "row_is_current": [True],
                "version_nbr": [3],
            }
        )
        result = parse_pjm_da(frame)
        self.assertEqual(result.loc[0, "market"], "PJM_DOM")
        self.assertEqual(result.loc[0, "da_lmp_usd_mwh"], 42.5)

    def test_nyiso_zone_j_parser(self) -> None:
        frame = pd.DataFrame(
            {
                "Time Stamp": ["09/01/2025 00:00"],
                "Name": ["N.Y.C."],
                "PTID": [61761],
                "LBMP ($/MWHr)": [35.0],
                "Marginal Cost Losses ($/MWHr)": [1.0],
                "Marginal Cost Congestion ($/MWHr)": [2.0],
            }
        )
        result = parse_nyiso_da(frame)
        self.assertEqual(
            result.loc[0, "interval_start_utc"].isoformat(),
            "2025-09-01T04:00:00+00:00",
        )
        self.assertEqual(result.loc[0, "congestion_component_usd_mwh"], 2.0)

    def test_nyiso_fall_dst_is_inferred_without_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "Time Stamp": [
                    "11/02/2025 00:00",
                    "11/02/2025 01:00",
                    "11/02/2025 01:00",
                    "11/02/2025 02:00",
                ],
                "PTID": [61761] * 4,
                "LBMP ($/MWHr)": [30.0] * 4,
                "Marginal Cost Losses ($/MWHr)": [0.0] * 4,
                "Marginal Cost Congestion ($/MWHr)": [0.0] * 4,
            }
        )
        result = parse_nyiso_da(frame)
        self.assertEqual(len(result), 4)
        self.assertTrue(result["interval_start_utc"].is_unique)

    def test_caiso_np15_parser_and_components(self) -> None:
        base = {
            "INTERVALSTARTTIME_GMT": "2025-09-01T00:00:00-00:00",
            "INTERVALENDTIME_GMT": "2025-09-01T01:00:00-00:00",
            "NODE": "TH_NP15_GEN-APND",
            "MARKET_RUN_ID": "DAM",
        }
        frame = pd.DataFrame(
            [
                {**base, "LMP_TYPE": "LMP", "MW": 49.35},
                {**base, "LMP_TYPE": "MCE", "MW": 45.00},
                {**base, "LMP_TYPE": "MCC", "MW": 3.00},
                {**base, "LMP_TYPE": "MCL", "MW": 1.35},
            ]
        )
        result = parse_caiso_da(frame)
        self.assertEqual(result.loc[0, "da_lmp_usd_mwh"], 49.35)
        self.assertEqual(result.loc[0, "energy_component_usd_mwh"], 45.0)
        with self.assertRaisesRegex(
            ValueError, "duplicate interval/LMP_TYPE"
        ):
            parse_caiso_da(pd.concat([frame, frame.iloc[[0]]]))

    def test_ercot_lz_north_parser(self) -> None:
        frame = pd.DataFrame(
            {
                "Delivery Date": ["09/01/2025"],
                "Hour Ending": ["01:00"],
                "Repeated Hour Flag": ["N"],
                "Settlement Point": ["LZ_NORTH"],
                "Settlement Point Price": [28.5],
            }
        )
        result = parse_ercot_da(frame)
        self.assertEqual(
            result.loc[0, "interval_end_utc"].isoformat(),
            "2025-09-01T06:00:00+00:00",
        )
        self.assertEqual(result.loc[0, "da_lmp_usd_mwh"], 28.5)

    def test_ercot_dst_transition_intervals_are_unique(self) -> None:
        frame = pd.DataFrame(
            {
                "Delivery Date": [
                    "03/09/2025",
                    "11/02/2025",
                    "11/02/2025",
                ],
                "Hour Ending": ["02:00", "02:00", "02:00"],
                "Repeated Hour Flag": ["N", "N", "Y"],
                "Settlement Point": ["LZ_NORTH"] * 3,
                "Settlement Point Price": [20.0, 30.0, 31.0],
            }
        )
        result = parse_ercot_da(frame)
        self.assertEqual(len(result), 3)
        self.assertTrue(result["interval_start_utc"].is_unique)
        fall = result.loc[
            result["da_lmp_usd_mwh"].isin([30.0, 31.0]),
            "interval_start_utc",
        ].sort_values()
        self.assertEqual(
            fall.iloc[1] - fall.iloc[0],
            pd.Timedelta(hours=1),
        )

    def test_ercot_document_selection_uses_discovery_revision(self) -> None:
        documents = [
            {
                "Document": {
                    "FriendlyName": "DAMLZHBSPP_2025",
                    "SecurityStatus": "P",
                    "PublishDate": "2026-01-01T08:00:00-06:00",
                    "DocID": "old",
                }
            },
            {
                "Document": {
                    "FriendlyName": "DAMLZHBSPP_2025",
                    "SecurityStatus": "P",
                    "PublishDate": "2026-02-01T08:00:00-06:00",
                    "DocID": "new",
                }
            },
        ]
        selected = select_ercot_document(
            documents, friendly_name="DAMLZHBSPP_2025"
        )
        self.assertEqual(selected["DocID"], "new")

    def test_ercot_sced_document_selection_uses_operating_date(self) -> None:
        documents = [
            {
                "Document": {
                    "ReportTypeID": "13052",
                    "FriendlyName": "60_Day_SCED_Disclosure",
                    "SecurityStatus": "P",
                    "Extension": "zip",
                    "PublishDate": "2025-10-31T04:40:00-05:00",
                    "DocID": "old",
                }
            },
            {
                "Document": {
                    "ReportTypeID": "13052",
                    "FriendlyName": "60_Day_SCED_Disclosure",
                    "SecurityStatus": "P",
                    "Extension": "zip",
                    "PublishDate": "2025-10-31T04:43:35-05:00",
                    "DocID": "new",
                }
            },
            {
                "Document": {
                    "ReportTypeID": "13052",
                    "FriendlyName": "60_Day_SCED_Disclosure",
                    "SecurityStatus": "P",
                    "Extension": "zip",
                    "PublishDate": "2025-11-01T04:43:35-05:00",
                    "DocID": "wrong-day",
                }
            },
        ]
        selected = select_ercot_sced_document(
            documents, sced_date="2025-09-01"
        )
        self.assertEqual(selected["DocID"], "new")

    def test_ercot_native_load_archive_resolves_repeated_hour(self) -> None:
        workbook = native_load_workbook(
            [
                ["11/02/2025 02:00", *["1"] * 8, "100"],
                ["11/02/2025 02:00 DST", *["2"] * 8, "200"],
            ]
        )
        result = read_ercot_native_load_xlsx(workbook)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["interval_start_utc"].is_unique)
        self.assertEqual(
            result.loc[0, "interval_start_utc"].isoformat(),
            "2025-11-02T06:00:00+00:00",
        )
        self.assertEqual(result.loc[0, "gross_demand_mw"], 200)
        self.assertEqual(
            result.loc[1, "interval_start_utc"].isoformat(),
            "2025-11-02T07:00:00+00:00",
        )
        self.assertEqual(result.loc[1, "gross_demand_mw"], 100)

    def test_ercot_sced_parser_resolves_repeated_hour(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "SCED Time Stamp": f"11/02/2025 01:00:00",
                    "Repeated Hour Flag": repeated,
                    "Resource Name": f"{resource_type}-{repeated}",
                    "Resource Type": resource_type,
                    "Telemetered Net Output ": value,
                }
                for repeated in ("N", "Y")
                for resource_type, value in (("WIND", 10), ("PVGR", 5))
            ]
        )
        result = parse_ercot_sced_executions(frame)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["timestamp_utc"].is_unique)
        self.assertEqual(
            result.loc[1, "timestamp_utc"] - result.loc[0, "timestamp_utc"],
            pd.Timedelta(hours=1),
        )

    def test_ercot_sced_hourly_is_duration_weighted(self) -> None:
        start = pd.Timestamp("2025-09-01T00:00:00Z")
        executions = pd.DataFrame(
            {
                "timestamp_utc": [
                    start - pd.Timedelta(minutes=5),
                    start + pd.Timedelta(minutes=5),
                    start + pd.Timedelta(minutes=35),
                    start + pd.Timedelta(hours=1),
                ],
                "wind_mw": [10, 20, 40, 50],
                "solar_mw": [0, 10, 30, 40],
                "wind_resource_count": [2, 2, 2, 2],
                "solar_resource_count": [1, 1, 1, 1],
            }
        )
        result = time_weight_ercot_sced_hourly(
            executions,
            start_utc=start,
            end_utc=start + pd.Timedelta(hours=1),
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "wind_mw"], 27.5)
        self.assertAlmostEqual(result.loc[0, "solar_mw"], 17.5)
        self.assertIn(
            "SCED_EXECUTION_GAP_GT_20_MINUTES",
            result.loc[0, "source_quality_flags"],
        )

    def test_ercot_renewable_comparison_never_calibrates(self) -> None:
        timestamps = pd.date_range(
            "2025-09-01T00:00:00Z", periods=3, freq="1h"
        )
        reconstructed = pd.DataFrame(
            {
                "interval_start_utc": timestamps,
                "wind_mw": [10.0, 20.0, 30.0],
                "solar_mw": [5.0, 6.0, 7.0],
            }
        )
        annual = pd.DataFrame(
            {
                "interval_start_utc": timestamps,
                "wind_mw": [9.0, 19.0, 29.0],
                "solar_mw": [5.0, 5.0, 5.0],
            }
        )
        result = compare_ercot_renewables(
            reconstructed,
            annual,
            start_utc=timestamps[0],
            end_utc=timestamps[-1] + pd.Timedelta(hours=1),
        )
        self.assertFalse(result["calibration_applied"])
        self.assertEqual(result["matched_hours"], 3)
        self.assertAlmostEqual(result["metrics"]["wind"]["bias_mw"], 1.0)

    def test_ercot_renewable_comparison_counts_shared_missing_hours(self) -> None:
        timestamps = pd.date_range(
            "2025-09-01T00:00:00Z", periods=3, freq="1h"
        )
        partial = pd.DataFrame(
            {
                "interval_start_utc": timestamps[:2],
                "wind_mw": [10.0, 20.0],
                "solar_mw": [5.0, 6.0],
            }
        )
        result = compare_ercot_renewables(
            partial,
            partial,
            start_utc=timestamps[0],
            end_utc=timestamps[-1] + pd.Timedelta(hours=1),
        )
        self.assertEqual(result["missing_sced_hours"], 1)
        self.assertEqual(result["missing_annual_hours"], 1)

    def test_miso_fixed_est_hour_ending_parser(self) -> None:
        rows = []
        for component, offset in (("LMP", 0.0), ("MCC", 1.0), ("MLC", -0.5)):
            row = {
                "Node": "MINN.HUB",
                "Type": "Hub",
                "Value": component,
            }
            row.update(
                {
                    f"HE {hour}": float(hour) + offset
                    for hour in range(1, 25)
                }
            )
            rows.append(row)
        result = parse_miso_da(pd.DataFrame(rows), "2025-09-01")
        self.assertEqual(len(result), 24)
        self.assertEqual(
            result.loc[0, "interval_start_utc"].isoformat(),
            "2025-09-01T05:00:00+00:00",
        )
        self.assertEqual(result.loc[23, "da_lmp_usd_mwh"], 24.0)
        self.assertEqual(result.loc[0, "congestion_component_usd_mwh"], 2.0)
        self.assertEqual(result.loc[0, "loss_component_usd_mwh"], 0.5)

    def test_spp_north_parser(self) -> None:
        frame = pd.DataFrame(
            {
                "Interval": ["09/01/2025 01:00:00"],
                "GMTIntervalEnd": ["09/01/2025 06:00:00"],
                "Settlement Location": ["SPPNORTH_HUB"],
                "Pnode": ["SPPNORTH_H"],
                "LMP": [25.0232],
                "MLC": [-0.4357],
                "MCC": [1.6213],
                "MEC": [23.8376],
            }
        )
        result = parse_spp_da(frame)
        self.assertEqual(
            result.loc[0, "interval_start_utc"].isoformat(),
            "2025-09-01T05:00:00+00:00",
        )
        self.assertAlmostEqual(
            result.loc[0, "loss_component_usd_mwh"], -0.4357
        )


if __name__ == "__main__":
    unittest.main()
