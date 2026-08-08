from __future__ import annotations

import unittest

import pandas as pd

from energy_model_v3.adapters.parsers import (
    parse_caiso_da,
    parse_ercot_da,
    parse_miso_da,
    parse_nyiso_da,
    parse_pjm_da,
    parse_spp_da,
)


class LockedSourceParserTests(unittest.TestCase):
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
