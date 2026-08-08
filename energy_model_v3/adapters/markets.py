"""Six locked market adapters and exact source semantics."""

from __future__ import annotations

import os
from typing import Any

from ..contract import SourceDescriptor
from .base import MarketAdapter


def _descriptor(
    *,
    market: str,
    source: str,
    feed: str,
    product: str,
    location: str,
    authentication: str,
    licensing: str,
    cadence: str,
    units: str,
    interval_semantics: str,
    source_timezone: str,
    dst_rule: str,
    status_field: str,
    revision_policy: str,
    query: dict[str, Any],
    redistribution: str,
    quality_flags: tuple[str, ...] = (),
) -> SourceDescriptor:
    return SourceDescriptor(
        market=market,
        source=source,
        feed=feed,
        product=product,
        location=location,
        authentication=authentication,
        licensing=licensing,
        cadence=cadence,
        units=units,
        interval_semantics=interval_semantics,
        source_timezone=source_timezone,
        dst_rule=dst_rule,
        status_field=status_field,
        revision_policy=revision_policy,
        query=query,
        redistribution=redistribution,
        quality_flags=quality_flags,
    )


class PJMAdapter(MarketAdapter):
    market = "PJM_DOM"
    credential = "PJM_API_KEY"
    redistribution = "code_query_metadata_hashes_and_derived_metrics_only"

    def descriptors(self) -> list[SourceDescriptor]:
        auth = "PJM_API_KEY subscription key"
        restricted = "code_query_metadata_hashes_and_derived_metrics_only"
        return [
            _descriptor(
                market=self.market,
                source="PJM Data Miner 2",
                feed="rt_da_monthly_lmps",
                product="settlement-final total_lmp_da",
                location="DOM zone pnode 34964545",
                authentication=auth,
                licensing="PJM data-use and redistribution restrictions apply",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_beginning",
                source_timezone="EPT",
                dst_rule="PJM EPT with explicit UTC conversion",
                status_field="is_verified/current/version",
                revision_policy="preserve versions and refresh trailing 90 days",
                query={
                    "pnode_id": 34964545,
                    "fields": ["total_lmp_da", "total_lmp_rt"],
                },
                redistribution=restricted,
            ),
            _descriptor(
                market=self.market,
                source="PJM Data Miner 2",
                feed="hrl_load_metered plus RTO wind/solar",
                product="PJM RTO physical stress tuple; DOM load diagnostic",
                location="PJM RTO physical / DOM price",
                authentication=auth,
                licensing="PJM data-use and redistribution restrictions apply",
                cadence="hourly",
                units="MW",
                interval_semantics="interval_ending",
                source_timezone="EPT",
                dst_rule="PJM EPT with explicit UTC conversion",
                status_field="is_verified/current/version",
                revision_policy="preserve versions and refresh trailing 90 days",
                query={
                    "gross": "PJM RTO gross load",
                    "renewables": ["PJM RTO wind", "PJM RTO solar"],
                    "diagnostic": "DOM hrl_load_metered",
                },
                redistribution=restricted,
                quality_flags=("price_physical_geography_mismatch_explicit",),
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        key = os.environ.get(self.credential or "", "")
        return (
            "https://api.pjm.com/api/v1/rt_da_monthly_lmps",
            {
                "params": {"rowCount": 1, "pnode_id": 34964545},
                "headers": {"Ocp-Apim-Subscription-Key": key},
            },
        )


class NYISOAdapter(MarketAdapter):
    market = "NYISO_NYC_J"
    redistribution = "raw_redistribution_caveat_metadata_required"

    def descriptors(self) -> list[SourceDescriptor]:
        common = {
            "market": self.market,
            "source": "NYISO MIS",
            "authentication": "public unauthenticated archive",
            "licensing": "NYISO archive terms; preserve correction hashes",
            "source_timezone": "America/New_York",
            "dst_rule": "resolve EST/EDT using America/New_York",
            "status_field": "archive correction/retrieval hash",
            "revision_policy": "retain every corrected archive hash",
            "redistribution": "raw_redistribution_caveat_metadata_required",
        }
        return [
            _descriptor(
                **common,
                feed="damlbmp",
                product="DAM zonal LBMP",
                location="N.Y.C. Zone J PTID 61761",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_beginning",
                query={"ptid": 61761, "archive": "monthly ZIP"},
            ),
            _descriptor(
                **common,
                feed="pal plus system fuel mix and BTM solar",
                product="zonal actual load plus system renewable context",
                location="Zone J load / NYCA wind and BTM solar",
                cadence="hourly canonical; RT retained at native 5-minute",
                units="MW",
                interval_semantics="interval_beginning",
                query={
                    "pal": "actual zonal load",
                    "wind": "system fuel mix",
                    "solar": "BTM estimate",
                    "net_load": "PAL directly; do not subtract BTM solar twice",
                },
                quality_flags=("renewable_geography_systemwide",),
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        return (
            "https://mis.nyiso.com/public/csv/damlbmp/"
            "20250901damlbmp_zone_csv.zip",
            {},
        )


class CAISOAdapter(MarketAdapter):
    market = "CAISO_NP15"
    redistribution = "attribution_and_terms_metadata_required"

    def descriptors(self) -> list[SourceDescriptor]:
        common = {
            "market": self.market,
            "authentication": "public unauthenticated endpoint",
            "licensing": "CAISO attribution and OASIS terms apply",
            "source_timezone": "UTC plus America/Los_Angeles display time",
            "dst_rule": "GMT field authoritative; blank/repeated display times retained",
            "status_field": "OASIS result/retrieval metadata",
            "revision_policy": "cache immutable response hashes; bounded 429 backoff",
            "redistribution": "attribution_and_terms_metadata_required",
        }
        return [
            _descriptor(
                **common,
                source="CAISO OASIS",
                feed="PRC_LMP v12",
                product="DAM LMP; LMP_TYPE=LMP",
                location="TH_NP15_GEN-APND",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_beginning",
                query={
                    "queryname": "PRC_LMP",
                    "version": 12,
                    "market_run_id": "DAM",
                    "node": "TH_NP15_GEN-APND",
                    "lmp_type": "LMP",
                    "timestamp": "INTERVALSTARTTIME_GMT",
                    "rt_sensitivity": "PRC_INTVL_LMP v3 RTM",
                },
            ),
            _descriptor(
                **common,
                source="CAISO Today's Outlook history",
                feed="netdemand.csv and fuelsource.csv",
                product="Current demand, published Net demand, Solar, Wind",
                location="CAISO system",
                cadence="5-minute native; hourly canonical",
                units="MW",
                interval_semantics="interval_beginning",
                query={
                    "net_load": "published Net demand used directly",
                    "no_double_subtract": True,
                },
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        return (
            "https://oasis.caiso.com/oasisapi/SingleZip",
            {
                "params": {
                    "queryname": "PRC_LMP",
                    "version": 12,
                    "market_run_id": "DAM",
                    "startdatetime": "20250901T00:00-0000",
                    "enddatetime": "20250901T01:00-0000",
                    "node": "TH_NP15_GEN-APND",
                    "resultformat": 6,
                }
            },
        )


class ERCOTAdapter(MarketAdapter):
    market = "ERCOT_LZ_NORTH"
    redistribution = "public_with_notices"

    def descriptors(self) -> list[SourceDescriptor]:
        common = {
            "market": self.market,
            "source": "ERCOT MIS",
            "authentication": "public MIS report discovery",
            "licensing": "public reuse with ERCOT notices",
            "source_timezone": "CST6CDT",
            "dst_rule": "parse delivery date/hour/interval/DSTFlag",
            "status_field": "DocID filename and publication metadata",
            "revision_policy": "retain all DocID filename hashes and revisions",
            "redistribution": "public_with_notices",
        }
        return [
            _descriptor(
                **common,
                feed="report 13060",
                product="annual DAM hourly settlement price",
                location="LZ_NORTH",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_ending",
                query={
                    "reportTypeId": 13060,
                    "discovery": "IceDocListJsonWS",
                    "download": "mirDownload?doclookupId=DocID",
                },
            ),
            _descriptor(
                **common,
                feed="reports 13101 and 13424",
                product="total system load and wind+solar physical tuple",
                location="ERCOT total physical / LZ_NORTH price",
                cadence="hourly",
                units="MW",
                interval_semantics="interval_ending",
                query={
                    "load_report": 13101,
                    "renewable_report": 13424,
                    "rt_sensitivity_report": 13061,
                    "forbidden_substitute": 12300,
                },
                quality_flags=("price_physical_boundary_mismatch_explicit",),
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        return (
            "https://www.ercot.com/misapp/servlets/IceDocListJsonWS",
            {"params": {"reportTypeId": 13060}},
        )


class MISOAdapter(MarketAdapter):
    market = "MISO_MINN_HUB"
    credential = "MISO_API_KEY"
    redistribution = "code_query_metadata_hashes_and_derived_metrics_only"

    def descriptors(self) -> list[SourceDescriptor]:
        restricted = "code_query_metadata_hashes_and_derived_metrics_only"
        return [
            _descriptor(
                market=self.market,
                source="MISO Market Reports",
                feed="YYYYMMDD_da_expost_lmp.csv",
                product="DA ex-post Hub LMP",
                location="MINN.HUB / Hub / LMP",
                authentication="public price report; physical tuple requires MISO_API_KEY",
                licensing="MISO restrictive redistribution terms apply",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_ending",
                source_timezone="fixed EST",
                dst_rule="fixed EST hour-ending, never civil-time inference",
                status_field="daily final report filename/hash",
                revision_policy="retain replaced daily report hashes",
                query={"row": ["MINN.HUB", "Hub", "LMP"]},
                redistribution=restricted,
            ),
            _descriptor(
                market=self.market,
                source="MISO Data Exchange",
                feed="historical load/fuel/wind/solar subscription",
                product="historical physical stress tuple",
                location="MISO system",
                authentication="MISO_API_KEY and subscription required",
                licensing="MISO restrictive redistribution terms apply",
                cadence="hourly",
                units="MW",
                interval_semantics="interval_ending",
                source_timezone="fixed EST",
                dst_rule="fixed EST hour-ending",
                status_field="Data Exchange version metadata",
                revision_policy="retain query version and response hash",
                query={
                    "current_display_api_acceptable": False,
                    "rt_sensitivity": "YYYYMMDD_rt_lmp_final.csv",
                },
                redistribution=restricted,
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        key = os.environ.get(self.credential or "", "")
        return (
            "https://api.misoenergy.org/MISORTWDDataBroker/"
            "DataBrokerServices.asmx",
            {"headers": {"Ocp-Apim-Subscription-Key": key}},
        )


class SPPAdapter(MarketAdapter):
    market = "SPP_NORTH_HUB"
    redistribution = "derived_metrics_only"

    def descriptors(self) -> list[SourceDescriptor]:
        common = {
            "market": self.market,
            "source": "SPP Marketplace Public Data",
            "authentication": "public unauthenticated file browser",
            "licensing": "no clear bulk redistribution license; derived metrics only",
            "source_timezone": "UTC for GenMix; market report fields normalized explicitly",
            "dst_rule": "UTC physical timestamps authoritative",
            "status_field": "path filename retrieval hash",
            "revision_policy": "freeze rolling-retention files with retrieval hash",
            "redistribution": "derived_metrics_only",
        }
        return [
            _descriptor(
                **common,
                feed="da-lmp-by-settlement-location",
                product="DAM hourly LMP",
                location="SPPNORTH_HUB",
                cadence="hourly",
                units="USD/MWh",
                interval_semantics="interval_beginning",
                query={
                    "path": "/YYYY/MM/By_Day/DA-LMP-SL-YYYYMMDDHH00.csv",
                    "rt_sensitivity": "rtbm-lmp-by-location",
                },
            ),
            _descriptor(
                **common,
                feed="GenMix365_SPP.csv / GenMixYTD",
                product="Load, Wind Market+Self, Solar Market+Self",
                location="SPP balancing authority",
                cadence="5-minute native; hourly canonical",
                units="MW",
                interval_semantics="interval_beginning",
                query={
                    "net_load": "load-wind-solar",
                    "rolling_retention": "365 days",
                    "detect_hourly_load_schema_changes": True,
                },
            ),
        ]

    def probe_url(self) -> tuple[str, dict[str, Any]]:
        return (
            "https://portal.spp.org/file-browser-api/download/"
            "da-lmp-by-settlement-location",
            {
                "params": {
                    "path": "/2025/09/By_Day/"
                    "DA-LMP-SL-202509010100.csv"
                }
            },
        )


ADAPTERS = {
    cls.market: cls
    for cls in (
        PJMAdapter,
        NYISOAdapter,
        CAISOAdapter,
        ERCOTAdapter,
        MISOAdapter,
        SPPAdapter,
    )
}


def get_adapter(market: str) -> MarketAdapter:
    try:
        return ADAPTERS[market]()
    except KeyError as exc:
        raise KeyError(f"unknown market adapter {market!r}") from exc
