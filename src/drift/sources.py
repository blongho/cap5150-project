"""Pinned data sources: KronoDroid real-device files and AOSP framework manifests."""

from __future__ import annotations

KRONODROID_BASE = "https://raw.githubusercontent.com/aleguma/kronodroid/main/real_device"
KRONODROID_FILES = {
    "real_malware_v1.zip": "03aa36a9c3aa3430523fcf87042659562e5af3219688562087107817095cb8cd",
    "real_legitimate_v1.zip": "21f6d507321856eefa6c40a31f484d3e532b5dea23144e3c2a9470161bd5f782",
}

AOSP_URL = (
    "https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/"
    "{tag}/core/res/AndroidManifest.xml"
)
# API level -> (AOSP release tag, release year)
API_LEVELS = {
    21: ("android-5.0.0_r1", 2014),
    22: ("android-5.1.0_r1", 2015),
    23: ("android-6.0.0_r1", 2015),
    24: ("android-7.0.0_r1", 2016),
    25: ("android-7.1.0_r1", 2016),
    26: ("android-8.0.0_r1", 2017),
    27: ("android-8.1.0_r1", 2017),
    28: ("android-9.0.0_r1", 2018),
    29: ("android-10.0.0_r1", 2019),
    30: ("android-11.0.0_r1", 2020),
    31: ("android-12.0.0_r1", 2021),
    32: ("android-12.1.0_r1", 2022),
    33: ("android-13.0.0_r1", 2022),
    34: ("android-14.0.0_r1", 2023),
}
AOSP_CHECKSUMS = "data/aosp_manifest_sha256.txt"
