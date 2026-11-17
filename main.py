"""SVAN 971 için etiketli eğitim ve yazılım testi simülasyonu."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from zoneinfo import ZoneInfo
from html import escape
from io import BytesIO
import json
from math import ceil, exp, floor, isfinite, log10, log2
from random import Random
import re
from secrets import token_hex
from struct import iter_unpack, pack
from unicodedata import normalize
from zipfile import ZIP_DEFLATED, ZipFile


FREQUENCIES = (
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
    500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500, 16000, 20000,
)
A_WEIGHT = (
    -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1,
    -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1,
    1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3,
)
C_WEIGHT = (
    -6.2, -4.4, -3, -2, -1.3, -0.8, -0.5, -0.3, -0.2, -0.1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.2, -0.3, -0.5, -0.8,
    -1.3, -2, -3, -4.4, -6.2, -8.5, -11.2,
)
REFERENCE_A_BANDS = (
    936, 936, 936, 936, 1047, 1072, 1248, 1652, 1914, 2070, 1737,
    1867, 2277, 3561, 3125, 3136, 2825, 2180, 2038, 1881, 2111,
    1854, 1883, 2005, 1650, 1327, 1113, 936, 936, 936, 936,
)
PERIODS = ("Türkiye Ek-2 – 5 dk referans", "ISO 1996-2 – elverişli veya geometri koşulu", "ISO 1996-2 – diğer koşullar")
DEFAULT_POINT_VARIATION_DB = 5.0
DEVICE_SERIALS = (107410, 107406, 23287)
UNAVAILABLE = 0xD000
STAT_LEVELS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90)
STAT_CLASSES = 120


@dataclass(frozen=True)
class Measurement:
    name: str
    start: datetime
    seconds: int
    a: float
    c: float
    event_count: int = 0
    event_seconds: int = 2
    event_gain: float = 25.0
    delay: int = 3
    high_range: bool = False
    event_min_gain: float | None = None
    natural_peaks: bool = False
    variable_baseline: bool = False
    variable_spectrum: bool = False
    spectrum_filter: str = "Z"
    reference_spectrum: bool = False
    detailed_svl: bool = False
    device_serial: int | None = None


@dataclass(frozen=True)
class Simulation:
    a: tuple[int, ...]
    c: tuple[int, ...]
    spectrum_offsets: tuple[int, ...]
    z_offset: int
    spectral_rows: bytes = b""
    band_average: tuple[int, ...] = ()
    average_z: int | None = None
    peak_starts: tuple[int, ...] = ()


def energy(level):
    return 10 ** (level / 10)


def average_centi(series):
    return round(1000 * log10(sum(energy(level / 100) for level in series) / len(series)))


def average_bands(data):
    return data.band_average or tuple(
        average_centi([max(0, a + offset) for a in data.a]) for offset in data.spectrum_offsets
    )


def weighted_energy(bins, weights):
    return sum(value * energy(weight) for value, weight in zip(bins, weights))


def spectral_shape(center, width):
    bins = [energy(-3 * (log2(freq / center) / width) ** 2) for freq in FREQUENCIES]
    a_energy = weighted_energy(bins, A_WEIGHT)
    normalized = [value / a_energy for value in bins]
    return normalized, weighted_energy(normalized, C_WEIGHT)


def spectral_endpoints(extended=False, reference=False):
    if reference:
        # Recover an approximate Z spectrum from the CVKALI22 A-weighted
        # band averages. Exclude the recurring display floor from the energy.
        floor_energy = energy(min(REFERENCE_A_BANDS) / 100)
        bins = [
            max(energy(level / 100) - floor_energy, 1e-12) / energy(weight)
            for level, weight in zip(REFERENCE_A_BANDS, A_WEIGHT)
        ]
        bass_fraction = [1 / (1 + (frequency / 140) ** 4) for frequency in FREQUENCIES]
        endpoints = []
        for fractions in ([1 - fraction for fraction in bass_fraction], bass_fraction):
            shaped = [value * fraction for value, fraction in zip(bins, fractions)]
            total_a = weighted_energy(shaped, A_WEIGHT)
            normalized = [value / total_a for value in shaped]
            endpoints.append((normalized, weighted_energy(normalized, C_WEIGHT)))
        return tuple(endpoints)
    return spectral_shape(1800, 0.95), spectral_shape(63 if extended else 125, 1.1)


def spectral_mix(c_minus_a, endpoints):
    (high, high_c), (low, low_c) = endpoints
    ratio = energy(c_minus_a)
    if not high_c <= ratio <= low_c:
        raise ValueError(
            f"Bu spektrum modeli için LCeq − LAeq farkı "
            f"{10 * log10(high_c):.2f} ile {10 * log10(low_c):.2f} dB arasında olmalı."
        )
    portion = (ratio - high_c) / (low_c - high_c)
    return [hi * (1 - portion) + lo * portion for hi, lo in zip(high, low)]


def spectrum_offsets(bins, weights):
    return (
        tuple(round(1000 * log10(value) + 100 * weight) for value, weight in zip(bins, weights)),
        round(1000 * log10(sum(bins))),
    )


def normal_profile(seconds, rng, fast_variation=0.0):
    # Interpolate between independent levels at irregular intervals, with
    # smaller second-to-second variation on top of the slower changes.
    levels = []
    previous = rng.uniform(-3.2, 3.2)
    while len(levels) < seconds:
        span = min(rng.randint(3, 15), seconds - len(levels))
        next_level = rng.uniform(-3.2, 3.2)
        levels.extend(
            previous + (next_level - previous) * (index + 1) / span
            + rng.uniform(-0.45, 0.45)
            for index in range(span)
        )
        previous = next_level
    if fast_variation:
        for index, level in enumerate(levels):
            deviation = rng.gauss(0.0, 4.5 if rng.random() < 0.15 else 2.0)
            levels[index] = level + fast_variation * max(-7.0, min(7.0, deviation))
    return levels


def add_peak_events(levels, count, duration, minimum_gain, maximum_gain, rng):
    # Sampling on the shortened timeline leaves a normal second between
    # events and at both ends after each full event is inserted.
    free_seconds = len(levels) - count * duration
    slots = sorted(rng.sample(range(1, free_seconds), count))
    starts = [slot + index * duration for index, slot in enumerate(slots)]
    event_energies = [
        sum(energy(level) for level in levels[start : start + duration])
        for start in starts
    ]
    minimum_factor = energy(minimum_gain)
    factors = [energy(rng.uniform(minimum_gain, maximum_gain)) for _ in starts]
    ordinary_energy = sum(map(energy, levels))
    minimum_total = ordinary_energy + sum(event_energies) * (minimum_factor - 1)
    extra_energy = sum(
        base * (factor - minimum_factor) for base, factor in zip(event_energies, factors)
    )
    # Prefer a modest LAeq correction for the normal intervals. Large or
    # frequent peaks may require a lower baseline even at the minimum gain.
    budget = max(ordinary_energy * energy(1.5), minimum_total * energy(0.4))
    scale = min(1.0, (budget - minimum_total) / extra_energy) if extra_energy else 1.0
    for start, factor in zip(starts, factors):
        gain = 10 * log10(minimum_factor + scale * (factor - minimum_factor))
        for index in range(start, start + duration):
            levels[index] += gain
    return tuple(starts)


def add_natural_peaks(levels, count, rng):
    # Reserve a normal sample before, after and between events.
    spare_seconds = len(levels) - 2 * count - 1
    durations = []
    for _ in range(count):
        duration = 2 if spare_seconds > 0 and rng.random() < 0.2 else 1
        durations.append(duration)
        spare_seconds -= duration - 1
    free_seconds = len(levels) - sum(durations)
    slots = sorted(rng.sample(range(1, free_seconds), count))
    occupied = 0
    starts = []
    for slot, duration in zip(slots, durations):
        start = slot + occupied
        starts.append(start)
        gain = rng.uniform(3.0, 8.0)
        levels[start] += gain
        if duration == 2:
            levels[start + 1] += gain * rng.uniform(0.45, 0.85)
        occupied += duration
    return tuple(starts)


def vary_spectrum(a, target_delta, peak_gain, rng, endpoints, band_weights, reference):
    (high, high_c), (low, low_c) = endpoints
    target_mix = (energy(target_delta) - high_c) / (low_c - high_c)
    a_energy = [energy(level / 100) for level in a]
    motion = []
    trend = peak_color = 0.0
    previous_gain = 0.0
    for gain in peak_gain:
        trend = 0.88 * trend + rng.uniform(-0.27, 0.27)
        if gain and not previous_gain:
            peak_color = rng.choice((-0.7, 0.0, 0.7))
        motion.append(trend + rng.uniform(-0.18, 0.18) + peak_color * min(1.0, gain / 5))
        previous_gain = gain

    amplitude = min(0.32, 0.10 + 0.85 * target_mix)
    # In the example, quiet A-weighted intervals retain more bass energy.
    bass_coupling = 0.025 if reference else 0.0
    raw_mix = [
        target_mix + amplitude * value - bass_coupling * ((level - a[0]) / 100)
        for level, value in zip(a, motion)
    ]
    # The A-energy-weighted mean of the bass fraction sets the total C/A
    # ratio. Recenter after clipping to keep the target throughout the file.
    target_energy = sum(a_energy) * target_mix
    lower, upper = -max(raw_mix), 1 - min(raw_mix)
    for _ in range(32):
        midpoint = (lower + upper) / 2
        mixed_energy = sum(
            weight * max(0.0, min(1.0, value + midpoint)) for weight, value in zip(a_energy, raw_mix)
        )
        if mixed_energy < target_energy:
            lower = midpoint
        else:
            upper = midpoint

    shift = (lower + upper) / 2
    packed = bytearray()
    c = []
    z = []
    band_energy = [0.0] * len(FREQUENCIES)
    for a_level, weight, value in zip(a, a_energy, raw_mix):
        mix = max(0.0, min(1.0, value + shift))
        bins = [weight * (hi * (1 - mix) + lo * mix) for hi, lo in zip(high, low)]
        bands = [
            max(0, round(1000 * log10(value) + 100 * correction))
            for value, correction in zip(bins, band_weights)
        ]
        c_level = round(1000 * log10(weight * (high_c * (1 - mix) + low_c * mix)))
        z_level = round(1000 * log10(sum(bins)))
        c.append(c_level)
        z.append(z_level)
        for index, level in enumerate(bands):
            band_energy[index] += energy(level / 100)
        packed += words([0, a_level, c_level, a_level, *bands, a_level, c_level, z_level])
    average = tuple(round(1000 * log10(value / len(a))) for value in band_energy)
    return tuple(c), bytes(packed), average, average_centi(z)


def simulate(config, rng=None, spectrum=None):
    if not 1 <= config.seconds <= 3600:
        raise ValueError("Bir kayıt 1 ile 3600 saniye arasında olmalı.")
    if not isfinite(config.a) or not isfinite(config.c):
        raise ValueError("LAeq ve LCeq sonlu sayı olmalı.")
    if config.spectrum_filter not in ("Z", "A"):
        raise ValueError("⅓ oktav spektrum filtresi Z veya A olmalı.")
    band_weights = A_WEIGHT if config.spectrum_filter == "A" else (0,) * len(FREQUENCIES)
    minimum_gain = config.event_min_gain
    if minimum_gain is None:
        minimum_gain = min(10.0, config.event_gain)
    if config.event_count < 0 or config.event_seconds < 1 or not 0 < minimum_gain <= config.event_gain <= 40:
        raise ValueError("Pik sayısını, süresini ve dB aralığını kontrol et. Alt sınır üst sınırı aşamaz.")
    occupied = config.event_count * (1 if config.natural_peaks else config.event_seconds)
    free = config.seconds - occupied
    if config.event_count and free < config.event_count + 1:
        raise ValueError("Pikler arasında ve kaydın iki ucunda en az birer normal saniye bırakılmalı.")

    rng = rng if rng is not None else Random()
    minimum, upper_peak = (30, 140) if config.high_range else (25, 123)
    fast_variation = 0.0
    if config.variable_baseline:
        room = min(config.a - minimum, upper_peak - max(config.a, config.c))
        fast_variation = max(0.0, min(1.0, (room - 4.0) / 7.0))
    relative = normal_profile(config.seconds, rng, fast_variation)
    if config.reference_spectrum:
        # The example has a wider central LAeq spread than the general profile.
        relative = [level * 1.55 for level in relative]
    regular = relative.copy() if config.variable_spectrum else ()
    peak_starts = ()
    if config.event_count:
        if config.natural_peaks:
            peak_starts = add_natural_peaks(relative, config.event_count, rng)
        else:
            peak_starts = add_peak_events(
                relative, config.event_count, config.event_seconds, minimum_gain, config.event_gain, rng,
            )
    normalization = 10 * log10(sum(map(energy, relative)) / config.seconds)
    a = tuple(round(100 * (config.a + level - normalization)) for level in relative)
    endpoints = spectral_endpoints(extended=True, reference=config.reference_spectrum)
    if spectrum is None:
        c_minus_a = config.c - config.a
        spectrum = spectral_mix(
            c_minus_a, endpoints if config.reference_spectrum else spectral_endpoints(),
        )
    else:
        c_minus_a = 10 * log10(weighted_energy(spectrum, C_WEIGHT) / weighted_energy(spectrum, A_WEIGHT))
    spectral_rows, band_average, average_z = b"", (), None
    if config.variable_spectrum:
        c, spectral_rows, band_average, average_z = vary_spectrum(
            a, c_minus_a, [level - base for level, base in zip(relative, regular)], rng,
            endpoints, band_weights, config.reference_spectrum,
        )
    else:
        c = tuple(level + round(100 * c_minus_a) for level in a)
    # The device checks underrange against integrated RMS. The reference
    # itself has individual seconds below the high range's 30 dB(A) rating.
    lower_level = average_centi(a) if config.reference_spectrum else min(a)
    if lower_level < minimum * 100 or max(max(a), max(c)) >= upper_peak * 100:
        raise ValueError("Senaryodaki seviyeler seçili cihaz aralığına sığmıyor. Seviyeleri veya aralığı düzenle.")
    offsets, z = spectrum_offsets(spectrum, band_weights)
    return Simulation(a, c, offsets, z, spectral_rows, band_average, average_z, peak_starts)


def words(values):
    return pack(f"<{len(values)}H", *values)


def record(payload):
    count = len(payload) + 2
    if count > 255:
        count += 2
        return words([0xC300, count, *payload, count, 0xCB00])
    return words([0xC300 | count, *payload, 0xCB00 | count])


def encoded_time(moment):
    if not 2000 <= moment.year <= 2127:
        raise ValueError("SVL tarihi 2000–2127 arasında olmalı.")
    date_code = (moment.year - 2000) * 512 + moment.month * 32 + moment.day
    time_code = moment.hour * 1800 + moment.minute * 30 + moment.second // 2
    return date_code, time_code


def slug(value):
    value = normalize("NFKD", value.replace("ı", "i").replace("İ", "I")).upper()
    return re.sub(r"[^A-Z0-9_-]", "", value)[:28]


def profile_settings(detailed):
    mask = 15 if detailed else 8
    return [0x1405, 0x0307, 0x0606, 1, 2, mask, 2, 0, 0x0606, 1, 3, mask, 3, 0, 0x0606, 0, 2, mask, 2, 0]


def centi_level(power):
    return max(0, round(1000 * log10(power)))


def statistical_level(histogram, bottom, exceeded):
    rank = sum(histogram) * (1 - exceeded / 100)
    before = 0
    for index, count in enumerate(histogram):
        if count and before + count >= rank:
            return round(100 * (bottom + index + (rank - before) / count))
        before += count


def period_level(start, series):
    # SVAN 971 day split 06–18, 18–22, 22–06 (Appendix D).
    totals, seconds = [0.0] * 3, [0] * 3
    first_second = start.hour * 3600 + start.minute * 60 + start.second
    for index, level in enumerate(series):
        hour = ((first_second + index) // 3600) % 24
        period = 0 if 6 <= hour < 18 else 1 if 18 <= hour < 22 else 2
        totals[period] += energy(level / 100)
        seconds[period] += 1
    active = [(index, count) for index, count in enumerate(seconds) if count]
    weights, penalties = (12, 4, 8), (1, energy(5), energy(10))
    combined = sum(weights[i] * penalties[i] * totals[i] / n for i, n in active)
    return centi_level(combined / sum(weights[i] for i, _ in active)), sum(1 << i for i, _ in active) << 5


def detailed_logger(config, samples, series):
    signals = tuple(zip(*series))
    rng = Random(f"{config.start.isoformat()}|{config.name}")
    bottom = min(18 if config.high_range else 13, max(0, min(min(s) for s in series) // 100 - 4))
    bottom = max(bottom, ceil(max(max(s) for s in series) / 100) - STAT_CLASSES + 12)
    histograms = [[0] * STAT_CLASSES for _ in series]
    current = [energy(level / 100) for level in signals[0]]
    detector = tuple(1 - exp(-0.1 / constant) for constant in (0.125, 0.035, 1.5))
    fast, attack, release = detector
    history = [[] for _ in series]
    band_extrema = [[65535] * 34, [0] * 34, [0] * 34]
    payload = bytearray()
    overload_seconds = 0
    overload_threshold = (14050 if config.high_range else 12350)
    for row, levels in zip(iter_unpack("<38H", samples), signals):
        modulation = [energy(rng.uniform(-3, 3)) for _ in range(10)]
        normalizer = sum(modulation) / len(modulation)
        factors = [value / normalizer for value in modulation]
        # Use a synthetic crest factor inside each subsecond window.
        crest = [rng.uniform(4, 11) for _ in factors]
        low = round(1000 * log10(min(factors)))
        high = round(1000 * log10(max(factors)))
        peak = round(max(1000 * log10(factor) + 100 * db for factor, db in zip(factors, crest)))
        overload = max(levels) + peak > overload_threshold
        overload_seconds += overload
        payload += words([row[0] | int(overload)])
        for profile, leq in enumerate(levels):
            power = energy(leq / 100)
            state = current[profile]
            low_power, high_power = float("inf"), 0.0
            for factor in factors:
                sample = power * factor
                alpha = fast if profile < 2 else attack if sample > state else release
                state += alpha * (sample - state)
                low_power = min(low_power, state)
                high_power = max(high_power, state)
                category = floor(log10(sample) * 10) - bottom
                histograms[profile][max(0, min(STAT_CLASSES - 1, category))] += 1
            current[profile] = state
            result = leq + peak, centi_level(high_power), centi_level(low_power)
            history[profile].append(result)
            payload += words([*result, leq])
        payload += words(row[4:])
        for index, level in enumerate(row[4:]):
            band_extrema[0][index] = min(band_extrema[0][index], max(0, level + low))
            band_extrema[1][index] = max(band_extrema[1][index], level + high)
            band_extrema[2][index] = max(band_extrema[2][index], level + peak)

    def takt(maxima, span):
        return average_centi([max(maxima[i:i + span]) for i in range(0, len(maxima), span)])

    extras = []
    for profile, readings in enumerate(history):
        peaks, maxima, minima = zip(*readings)
        den, flags = period_level(config.start, series[profile])
        extras.append((max(peaks), max(maxima), min(minima), centi_level(current[profile]),
                       den, takt(maxima, 3), takt(maxima, 5), flags))
    ln = [0x2B17, 0x0307, len(STAT_LEVELS)]
    for exceeded in STAT_LEVELS:
        ln.extend([exceeded, *(statistical_level(hist, bottom, exceeded) for hist in histograms)])
    histogram = []
    for profile, counts in enumerate(histograms):
        histogram.extend([(1 << (8 + profile)) | 0x0B, 2 * STAT_CLASSES + 2])
        for count in counts:
            histogram.extend((count & 0xFFFF, count >> 16))
    stat_header = words([0x0E09, 0x0307, *([0x040A, STAT_CLASSES, bottom * 10, 10] * 3)])
    return payload, extras, band_extrema, stat_header, record(ln), record(histogram), overload_seconds


def svl(config, data, file_stem=None):
    """Write the SVAN 971 1/3-octave logger format from App. B of the manual."""
    start_date, start_time = encoded_time(config.start)
    device_filename = slug(file_stem or config.name)[:8].encode("ascii")
    # The reference encodes an empty description as a space followed by NUL padding.
    description = b" ".ljust(130, b"\0")

    pc_header = b"SvanPC" + words([26, 32, 71] + [0] * 10)
    file_header = words([0x0E01]) + device_filename.ljust(8, b" ") + words(
        [0x0300, start_date, start_time] + [0] * 6
    )
    serial = config.device_serial or 0
    # SVAN 971 unit header stores the serial in two 16-bit words.
    unit = words([0x0B02, serial & 0xFFFF, 971, 114, 0, 1, 1, 114, 114, 1, serial >> 16])
    calibration = words([0x0B47, 0, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 9400, 0xFFFF])
    user_text = words([0x4203]) + description
    general = [
        0x4004, start_date, start_time, 3, 2, 2 if config.high_range else 1, 0, 1, 1, 3,
        config.delay, config.seconds, 0, 2, 0, 2 if config.spectrum_filter == "A" else 1,
        8, 480, 0, 1, 0, 0, 800, 0, 3,
        800, 0, 3, 800, 0, 5, 127, 0, 0, 0, 0,
        *([1150] * 6), *([0xFFFF] * 7), 0,
        0x0A00, 0x0A1E, 0, 0x0B00, 0x0B1E, 0, 0x0C00, 0x0D00, 0,
        0x0E00, 0x0E1E, 0, 0x0F00, 0x0F1E,
    ]
    # Profile, trigger and display descriptors follow the 971 filesystem revision.
    triggers = [
        0x0E2B, 0, 0, 1000, 100, 0, 0, 0, 5, 0, 0, 0, 0, 0,
        0x0E2C, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0E31, 0, 0, 1000, 100, 0, 0, 1200, 10, 0, 1, 16, 15441, 0,
        0x0E2D, 0, 0, 0, 0, 0, 0, 4800, 0, 0, 0, 16, 15441, 0,
    ]
    display = [
        0x1F48, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
    ]
    # Logger records contain flags, three Leq profiles, 31 bands and A/C/Z totals.
    samples = data.spectral_rows
    if not samples:
        samples = bytearray()
        for a, c in zip(data.a, data.c):
            bands = [max(0, a + offset) for offset in data.spectrum_offsets]
            samples += words([0, a, c, a, *bands, a, c, a + data.z_offset])

    profiles = (data.a, data.c, data.a)
    stat_header, stat_levels, histogram = b"", b"", b""
    band_extrema = []
    overload = 0
    details = [(UNAVAILABLE,) * 4 + (0,) * 4] * len(profiles)
    if config.detailed_svl:
        samples, details, band_extrema, stat_header, stat_levels, histogram, overload = detailed_logger(
            config, samples, profiles,
        )
        general[6] = 2 if overload else 0
        general[31] = 127  # main results, four spectrum blocks, levels and histograms

    main_results = [0x4407, 0x0307]
    for profile, (series, detail) in enumerate(zip(profiles, details)):
        leq = average_centi(series)
        peak, maximum, minimum, spl, den, takt3, takt5, flags = detail
        # The second profile's time slot carries the overload duration.
        elapsed = (config.seconds, overload, 0)[profile]
        main_results.extend([
            0x1608, elapsed, 0, peak, round(leq + 1000 * log10(config.seconds)),
            maximum, minimum, spl, leq, den, takt3, takt5, *([0] * 7),
            flags | (2 if overload else 0),
            UNAVAILABLE, UNAVAILABLE,
        ])
    spectrum = [0x2710, 0x0101, 2000, 31, 3]
    spectrum += average_bands(data)
    total_z = data.average_z if data.average_z is not None else average_centi(data.a) + data.z_offset
    spectrum += [average_centi(data.a), average_centi(data.c), total_z]
    for tag, spectrum_extreme in zip((0x2728, 0x2729, 0x2732), band_extrema):
        spectrum.extend([tag, 0x0101, 2000, 31, 3, *spectrum_extreme])
    summary = record([0x0359, 2, 0]) + record(main_results) + stat_levels + record(spectrum) + histogram
    logger_size = len(samples) + len(summary)
    logger = words([0x0E0F, 1, 0, 2000, 31, 3]) + pack(
        "<IIII", logger_size, config.seconds, config.seconds, 0
    )
    return b"".join((
        pc_header, file_header, unit, calibration, user_text, words(general), words(triggers),
        words(profile_settings(config.detailed_svl)), words(display), stat_header, logger, samples, summary,
    ))


def run_app():
    import streamlit as st
    from streamlit.components.v1 import html

    st.set_page_config(page_title="Gürültü Saha Asistanı", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)
    storage = st.components.v2.component("noise_workspace_storage", js=WORKSPACE_STORAGE)
    with st.container(key="noise-storage"):
        storage_slot = st.empty()
    if "noise_workspace" not in st.session_state:
        with storage_slot.container():
            result = storage(data={"read": True}, key="noise_workspace_bridge",
                             on_loaded_change=lambda: None, on_failed_change=lambda: None)
        if result.loaded is None:
            st.caption("Son çalışma yükleniyor…")
            return
        restore_workspace(st, result.loaded)
        st.rerun()

    html(DRAWER, height=0)
    render_workspace(st)
    with storage_slot.container():
        storage(data={"workspace": json.dumps(st.session_state.noise_workspace, default=lambda value: value.isoformat())},
                key="noise_workspace_bridge", on_loaded_change=lambda: None, on_failed_change=lambda: None)
    if st.session_state.get("noise_workspace_bridge", {}).get("failed"):
        st.warning("Tarayıcı çalışma kaydını saklayamıyor.")


def restore_workspace(st, raw):
    try:
        workspace = json.loads(raw)
        if not isinstance(workspace.get("draft"), dict):
            raise ValueError("Invalid draft")
    except (AttributeError, TypeError, ValueError):
        workspace = {"draft": {}, "run": None}
    st.session_state.noise_workspace = workspace
    saved_run = workspace.get("run")
    if saved_run:
        try:
            inputs = dict(saved_run["inputs"])
            inputs["day"] = date.fromisoformat(inputs["day"])
            inputs["clock"] = time.fromisoformat(inputs["clock"])
            st.session_state.results = build_results(**inputs, seed=saved_run["seed"])
        except (KeyError, TypeError, ValueError, OverflowError):
            workspace["run"] = None
            st.session_state.noise_restore_failed = True


def reset_workspace(st):
    st.session_state.noise_workspace = {"draft": {}, "run": None}
    for key in list(st.session_state):
        if key.startswith("noise_field_") or key in (
            "device_selected", "results", "noise_defaults", "noise_restore_failed",
        ):
            del st.session_state[key]


def work_field(st, widget, key, label, default=None, *, multiple=False, **options):
    draft = st.session_state.noise_workspace["draft"]
    defaults = st.session_state.setdefault("noise_defaults", {})
    widget_key = f"noise_field_{key}"
    if widget_key not in st.session_state:
        saved = draft.get(key, default)
        if isinstance(saved, str) and isinstance(default, (date, time)):
            try:
                saved = type(default).fromisoformat(saved)
            except ValueError:
                saved = default
        defaults[key] = saved
    selected = defaults[key]
    if multiple:
        options["default"] = [item for item in selected if item in options["options"]]
    elif "options" in options:
        choices = options["options"]
        options["index"] = choices.index(selected) if selected in choices else choices.index(default)
    else:
        options["value"] = selected
    value = widget(label, key=widget_key, **options)
    draft[key] = value
    return value


def render_workspace(st):
    draft = st.session_state.noise_workspace["draft"]
    with st.sidebar:
        st.markdown("## Gürültü Saha Asistanı")
        page = work_field(st, st.radio, "page", "Menü", "Ana ekran",
                          options=("Ana ekran", "Ölçüm oluştur", "Sonuçlar"), label_visibility="collapsed")
        st.caption("SVAN 971 · Simülasyon")
        st.button("Sıfırla", key="reset_workspace", on_click=reset_workspace, args=(st,),
                  help="Bu tarayıcıdaki formu ve son simülasyon sonucunu temizler.")

    st.markdown(
        '<header class="noise-hero"><h1>Gürültü Saha Asistanı</h1>'
        '<p>Svantek SVAN 971 · Simülasyon</p></header>',
        unsafe_allow_html=True,
    )
    if st.session_state.pop("noise_restore_failed", False):
        st.warning("Son sonuç geri yüklenemedi. Kayıtlı formdan yeni simülasyon oluşturabilirsin.")
    if page == "Sonuçlar":
        show_results(st)
        return
    if page == "Ana ekran" and not draft.get("device_selected"):
        with st.container(border=True):
            st.subheader("SVAN 971")
            st.caption("Svantek · Çevresel gürültü")
            st.button("SVAN 971’i seç", type="primary", use_container_width=True,
                      on_click=lambda: draft.update(device_selected=True))
        return

    inputs = measurement_form(st)
    saved_run = st.session_state.noise_workspace.get("run")
    previous_limit = saved_run["inputs"].get("maximum_peaks") if saved_run else None
    limit_changed = saved_run is not None and previous_limit != inputs["maximum_peaks"]
    filter_changed = saved_run is not None and (
        saved_run["inputs"].get("spectrum_filter", "Z") != inputs["spectrum_filter"]
    )
    if st.button("Ölçüm Yap", type="primary", use_container_width=True) or limit_changed or filter_changed:
        try:
            seed = token_hex(16)
            st.session_state.results = build_results(**inputs, seed=seed)
            st.session_state.noise_workspace["run"] = {"inputs": inputs, "seed": seed}
            draft.pop("selected_record", None)
            st.session_state.pop("noise_field_selected_record", None)
            if filter_changed:
                st.success(f"{inputs['spectrum_filter']} spektrum filtresi uygulandı. Noktaların sonuçları yenilendi.")
            elif limit_changed:
                st.success(f"Pik üst sınırı {inputs['maximum_peaks']} uygulandı. Noktaların sonuçları yenilendi.")
        except ValueError as error:
            st.error(str(error))
            return
    if st.session_state.get("results"):
        show_results(st)


def section(st, label):
    st.markdown(f'<div class="noise-section"><h3>{escape(label)}</h3></div>', unsafe_allow_html=True)


def measurement_form(st):
    field = partial(work_field, st)
    section(st, "Ölçüm metodu")
    period = field(st.selectbox, "period", "TS ISO 1996-1/2 · Çevresel gürültü", PERIODS[0], options=PERIODS)
    section(st, "Tesis ve ölçüm noktası")
    plant = field(st.text_input, "plant", "Tesis adı", "").strip()
    point_count = field(st.number_input, "point_count", "Ölçüm noktası sayısı", 1, min_value=1, step=1)
    point = f"{plant}1"
    following_points = []
    if point_count > 1:
        st.caption("Her geçiş süresi, önceki noktadaki son kayıt bittiğinde başlar.")
        for number in range(2, point_count + 1):
            transition = field(
                st.number_input, f"point_transition_{number}",
                f"Nokta {number - 1} → {number} geçiş süresi (sn)", 0, min_value=0, step=1,
            )
            following_points.append((f"{plant}{number}", transition))
    setting = field(st.selectbox, "setting", "Ortam", "Dış ortam", options=("Dış ortam", "İç ortam"))
    mode = field(st.text_input, "mode", "İşletme modu", "", placeholder="Örn. tesis tam yükte")
    with st.expander("Saha bilgileri"):
        placement = field(st.selectbox, "placement", "Mikrofon konumu", "Belirtilmedi",
                          options=("Belirtilmedi", "Serbest alan", "Cephe önü", "Yüzeye monteli", "Oda içi"))
        height = field(st.number_input, "height", "Mikrofon yüksekliği (m)", min_value=0.0, step=0.1)
        distance = field(st.number_input, "distance", "Kaynağa yatay uzaklık (m)", min_value=0.0, step=0.5)
        weather = field(st.text_input, "weather", "Hava koşulları ve saha notu", "", placeholder="Rüzgâr hızı/yönü, sıcaklık, nem ve yağış")
    site_notes = " · ".join(filter(None, (
        placement if placement != "Belirtilmedi" else "",
        f"Mikrofon {height:g} m" if height is not None else "",
        f"Kaynak uzaklığı {distance:g} m" if distance is not None else "",
        weather.strip(),
    )))

    section(st, "Tarih, saat ve süre")
    now = datetime.now(ZoneInfo("Europe/Istanbul"))
    draft = st.session_state.noise_workspace["draft"]
    if "clock_hms" not in draft and isinstance(draft.get("clock"), (str, time)):
        try:
            old_clock = draft["clock"]
            if isinstance(old_clock, str):
                old_clock = time.fromisoformat(old_clock)
            draft["clock_hms"] = old_clock.replace(
                second=draft.get("start_second", old_clock.second),
            ).strftime("%H:%M:%S")
        except (TypeError, ValueError):
            pass
    with st.container(key="datetime"):
        date_col, time_col = st.columns(2)
        day = field(date_col.date_input, "day", "Başlangıç tarihi", now.date(), format="DD.MM.YYYY")
        clock_text = field(
            time_col.text_input, "clock_hms", "Başlangıç saati (HH:MM:SS)", now.strftime("%H:%M:%S"),
            placeholder="14:25:22",
            help="Saat, dakika ve saniyeyi gir. SVL başlığı tek saniyeleri bir önceki çift saniyeye kaydeder.",
        ).strip()
    clock = None
    if re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]", clock_text):
        clock = time.fromisoformat(clock_text)
    default_minutes = (5, 10, 30)[PERIODS.index(period)]
    period_min = field(st.number_input, "period_min", "Kaynağın tam çalışma döngüsü (dk, biliniyorsa)", 0, min_value=0, max_value=60)
    recommended = max(default_minutes, period_min)
    manual = field(st.toggle, "manual_duration", "Ölçüm süresini manuel belirle", False)
    if manual:
        minutes = field(st.number_input, "duration", "Ölçüm süresi (dk)", recommended, min_value=1, max_value=60)
    else:
        minutes = recommended
        st.caption(f"Örnek kayıt süresi {minutes} dk")
    if minutes < recommended:
        st.warning(f"Seçilen saha planında önerilen süre {recommended} dk. Simülasyon oluşturulabilir.")
    repeats = field(st.number_input, "repeats", "Cihaz tekrar sayısı", 1, min_value=1, max_value=10)
    gap_seconds = field(
        st.number_input, "gap", "Aynı noktadaki ölçümler arası bekleme (sn)", 0, min_value=0, step=1,
        help="Aynı noktadaki tekrarlar ve arka plan kaydı için kullanılır.",
    )

    section(st, "Kayıt ve ses seviyesi")
    name = field(st.text_input, "record_name", "SVL kayıt adı", "ORCALI4", max_chars=28)
    a = field(st.number_input, "a", "Kaynak açık · Referans LAeq [dB(A)]", 60.0,
              min_value=25.0, max_value=139.0, step=0.01)
    a_bounds = (a, a)
    if point_count > 1:
        low_col, high_col = st.columns(2)
        a_low = field(low_col.number_input, "a_low", "LAeq alt sınırı [dB(A)]",
                      min_value=25.0, max_value=139.0, step=0.01)
        a_high = field(high_col.number_input, "a_high", "LAeq üst sınırı [dB(A)]",
                       min_value=25.0, max_value=139.0, step=0.01)
        a_bounds = (
            max(25.0, round(a - DEFAULT_POINT_VARIATION_DB, 2)) if a_low is None else a_low,
            min(139.0, round(a + DEFAULT_POINT_VARIATION_DB, 2)) if a_high is None else a_high,
        )
        st.caption(f"Boş bırakılan sınırlar için referansın ±{DEFAULT_POINT_VARIATION_DB:g} dB çevresi kullanılır.")
    manual_targets = []
    if field(st.checkbox, "set_point_targets", "Nokta bazında manuel hedef belirle", False):
        selected = field(
            st.multiselect, "manual_points", "Manuel hedef verilecek noktalar", [],
            options=range(1, point_count + 1), multiple=True,
            format_func=lambda number: f"{number}. nokta · {plant}{number}",
        )
        for number in sorted(set(selected).intersection(range(1, point_count + 1))):
            manual_a = field(st.number_input, f"point_a_{number}", f"{plant}{number} · Nihai LAeq [dB(A)]", a,
                             min_value=25.0, max_value=139.0, step=0.01)
            manual_c = field(st.number_input, f"point_c_{number}", f"{plant}{number} · Nihai LCeq [dB(C)]",
                             min_value=-10.0, max_value=145.0, step=0.01, placeholder="Otomatik")
            manual_targets.append((number, manual_a, manual_c))
    previous_count = draft.get("peak_count", 0) if draft.get("character") == "Pikli" else 0
    maximum_peaks = field(
        st.number_input, "maximum_peaks", "Saniyelik grafikte maksimum yükseliş sayısı", previous_count,
        min_value=0, step=1,
        help="Her kaynak açık kaydı için sayı sıfır ile bu sınır arasında ayrı seçilir. "
             "Sonuç varken bu sınırı değiştirirsen sonuçlar otomatik yenilenir. "
             "Yükselişler yaklaşık 3–8 dB olur. Çoğu 1 saniye sürer. "
             "Cihazın anlık Lpeak değeri ayrı bir ölçüdür.",
    )

    with st.expander("Arka plan ve cihaz ayarları"):
        device_serial = field(
            st.selectbox, "device_serial", "Cihaz seri numarası", None, options=(None, *DEVICE_SERIALS),
            format_func=lambda serial: "Seri numarası seç" if serial is None else str(serial),
        )
        manual_c = field(st.checkbox, "manual_c", "Genel LCeq referansı belirle", False)
        c = field(st.number_input, "c", "Referans LCeq [dB(C)]", min(145.0, a + 7.0),
                  min_value=-10.0, max_value=145.0, step=0.01) if manual_c else None
        c_bounds = None
        if field(st.checkbox, "limit_c", "Otomatik LCeq için sınır belirle", False):
            low_col, high_col = st.columns(2)
            c_low = field(low_col.number_input, "c_low", "LCeq alt sınırı [dB(C)]",
                          min_value=-10.0, max_value=145.0, step=0.01)
            c_high = field(high_col.number_input, "c_high", "LCeq üst sınırı [dB(C)]",
                           min_value=-10.0, max_value=145.0, step=0.01)
            c_bounds = (c_low, c_high)
        has_background = field(st.checkbox, "has_background", "Kaynak kapalıyken arka plan kaydı ekle", False)
        background = field(st.number_input, "background", "Arka plan hedef LAeq [dB(A)]", 45.0,
                           min_value=25.0, max_value=139.0, step=0.1) if has_background else None
        high_range = field(
            st.selectbox, "high_range", "Cihaz aralığı", False, options=(False, True),
            format_func=lambda high: "Yüksek · 30 dB(A) Leq / 140 dB Peak"
            if high else "Düşük · 25 dB(A) Leq / 123 dB Peak",
            help="SVAN 971’in alt Leq sınırı ve üst anlık tepe sınırı. "
                 "Yüksek aralık, daha yüksek tepelere yer açar.",
        )
        delay = field(st.number_input, "delay", "Başlangıç gecikmesi (sn)", 3, min_value=0, max_value=60,
                                help="Cihaz ayarı olarak kaydedilir. Formdaki başlangıç, kaydın ilk zamanıdır.")
        spectrum_filter = field(
            st.selectbox, "spectrum_filter", "⅓ oktav spektrum filtresi", "Z", options=("Z", "A"),
            format_func=lambda weighting: "A · A ağırlıklı" if weighting == "A" else "Z · Ağırlıksız",
            help="Bant grafiğine ve SVL’deki ⅓ oktav bantlarına uygulanır. "
                 "Sonuç varken filtreyi değiştirirsen sonuçlar otomatik yenilenir.",
        )
        reference_spectrum = field(
            st.checkbox, "reference_spectrum", "CVKALI22 örneğine benzer frekans dağılımı", False,
            help="Kaynak açık simülasyonunda örnek kaydın ortalama bant biçimini esas alır. "
                 "Örnek dosya A ağırlıklı ⅓ oktav bantlarıyla kaydedilmiştir.",
        )
        st.caption("Kayıt adımı 1 sn · ⅓ oktav · Başlangıç senkronizasyonu kapalı")
    return dict(
        plant=plant, point=point, setting=setting, mode=mode, day=day, clock=clock,
        site_notes=site_notes,
        minutes=minutes, repeats=repeats, name=name, a=a, c=c, count=0,
        length=1, gain=8.0, background=background, high_range=high_range, delay=delay,
        maximum_peaks=maximum_peaks, variable_baseline=True, variable_spectrum=True,
        spectrum_filter=spectrum_filter, reference_spectrum=reference_spectrum, detailed_svl=True,
        device_serial=device_serial,
        gap_seconds=gap_seconds, following_points=following_points,
        a_bounds=a_bounds, c_bounds=c_bounds, manual_targets=manual_targets,
    )


def point_targets(a, c, a_bounds, c_bounds, manual, single_point, endpoints, rng):
    min_delta = ceil(1000 * log10(endpoints[0][1])) / 100
    max_delta = floor(1000 * log10(endpoints[1][1])) / 100
    a_low, a_high = a_bounds
    c_low, c_high = c_bounds if c_bounds else (-10.0, 145.0)
    reference_delta = c - a if c is not None else 6.0
    if manual:
        point_a, point_c = manual
        if point_c is not None:
            return point_a, point_c, 0.35
    else:
        if single_point and c is not None and c_bounds is None:
            return a, c, 0.35
        a_low = max(a_low, round(c_low - max_delta, 2))
        a_high = min(a_high, round(c_high - min_delta, 2))
        if a_low > a_high:
            raise ValueError("LAeq ve LCeq aralıkları seçili frekans modeliyle uyuşmuyor.")
        point_a = round(rng.triangular(a_low, a_high, min(a_high, max(a_low, a))), 2)

    delta_low = max(min_delta, round(c_low - point_a, 2))
    delta_high = min(max_delta, round(c_high - point_a, 2))
    if c_bounds is None:
        delta_low, delta_high = max(delta_low, min(0.0, delta_high)), min(delta_high, max(15.0, delta_low))
    if delta_low > delta_high:
        raise ValueError("Bu LAeq hedefiyle belirtilen LCeq aralığında sonuç üretilemiyor.")
    delta = rng.triangular(delta_low, delta_high, min(delta_high, max(delta_low, reference_delta)))
    point_c = round(point_a + delta, 2)
    margin = min(0.35, point_c - c_low, c_high - point_c)
    if not manual and not single_point:
        margin = min(margin, point_a - a_low, a_high - point_a)
    return point_a, point_c, max(0.0, margin)


def repeat_offsets(repeats, margin, rng):
    # Pair equal and opposite energy changes so the combined Leq stays at target.
    offsets = [0.0] if repeats % 2 else []
    limit = 1 - energy(-margin)
    for _ in range(repeats // 2):
        change = rng.uniform(limit / 4, limit)
        offsets.extend((10 * log10(1 - change), 10 * log10(1 + change)))
    rng.shuffle(offsets)
    return offsets


def build_results(plant, point, setting, mode, day, clock, minutes, repeats, name,
                  a, c, count, length, gain, background, high_range, delay, site_notes="",
                  min_gain=None, gap_seconds=0, following_points=(), seed=None,
                  a_bounds=None, c_bounds=None, manual_targets=(), maximum_peaks=None,
                  variable_baseline=False, variable_spectrum=False, spectrum_filter="Z",
                  reference_spectrum=False, detailed_svl=False, device_serial=None):
    if not isinstance(clock, time):
        raise ValueError("Başlangıç saatini saat:dakika:saniye biçiminde gir. Örnek 14:25:22.")
    record_name = slug(name)
    if not plant.strip() or not point.strip() or not record_name.strip("_-"):
        raise ValueError("Tesis adı, ölçüm noktası ve SVL kayıt adı gerekli.")
    if device_serial is not None and device_serial not in DEVICE_SERIALS:
        raise ValueError("Listeden geçerli bir cihaz seri numarası seç.")
    if gap_seconds < 0:
        raise ValueError("Ölçümler arası bekleme süresi negatif olamaz.")
    points = [(point.strip(), 0)]
    for number, (next_point, transition) in enumerate(following_points, start=2):
        if not next_point.strip():
            raise ValueError(f"{number}. noktanın adını gir.")
        if transition < 0:
            raise ValueError(f"Nokta {number - 1} → {number} geçiş süresi negatif olamaz.")
        points.append((next_point.strip(), transition))
    multi_point = len(points) > 1
    overrides = {number: (level_a, level_c) for number, level_a, level_c in manual_targets}
    if a_bounds is not None and not a_bounds[0] <= a <= a_bounds[1]:
        raise ValueError("Referans LAeq, alt ve üst LAeq sınırlarının arasında olmalı.")
    if c_bounds is not None and (None in c_bounds or c_bounds[0] > c_bounds[1]):
        raise ValueError("LCeq için geçerli bir alt ve üst sınır gir.")
    # Older saved sessions replay the original seeded generator and spectrum.
    endpoints = spectral_endpoints(extended=True, reference=reference_spectrum) if a_bounds is not None else None
    plant = plant.strip()
    title = f"{plant} · {len(points)} ölçüm noktası" if multi_point else f"{plant} / {point.strip()}"
    next_start = datetime.combine(day, clock)
    duration = timedelta(minutes=minutes)
    if maximum_peaks is not None and maximum_peaks < 0:
        raise ValueError("Maksimum pik sayısı negatif olamaz.")
    peak_limit = min(maximum_peaks, (minutes * 60 - 1) // 2) if maximum_peaks is not None else None
    record_interval = duration + timedelta(seconds=gap_seconds)
    rows = []
    rng = Random(seed)
    record_count = len(points) * (repeats + (background is not None))
    sequence_width = max(2, len(str(record_count)))
    for number, (point_name, transition) in enumerate(points, start=1):
        next_start += timedelta(seconds=transition)
        point_a, point_c = a, c
        spectrum = None
        shifts = [0.0] * repeats
        if endpoints is not None:
            try:
                point_a, point_c, margin = point_targets(
                    a, c, a_bounds, c_bounds, overrides.get(number), not multi_point, endpoints, rng,
                )
                spectrum = spectral_mix(round(point_c - point_a, 2), endpoints)
                shifts = repeat_offsets(repeats, margin, rng)
            except ValueError as error:
                raise ValueError(f"{point_name} · {error}") from error
        for index in range(repeats + (background is not None)):
            is_background = index == repeats
            label = "Arka plan" if is_background else f"Kaynak açık · Tekrar {index + 1}"
            suffix = "FON" if is_background else f"{index + 1:02d}"
            if multi_point:
                label = f"{number}. {point_name} · {label}"
                suffix = f"N{number:03d}_{suffix}"
                device_id = f"{len(rows) + 1:05d}"
            else:
                device_id = record_name[:max(1, 5 - len(suffix))] + suffix
            level_a = background if is_background else point_a + shifts[index]
            level_c = background + 7.0 if is_background else point_c + shifts[index]
            event_count = 0 if is_background else count
            if not is_background and peak_limit is not None:
                event_count = rng.randint(0, peak_limit) if peak_limit else 0
            config = Measurement(device_id, next_start + record_interval * index,
                                 minutes * 60, level_a, level_c, event_count,
                                 length, gain, delay, high_range, event_min_gain=min_gain,
                                 natural_peaks=peak_limit is not None, variable_baseline=variable_baseline,
                                 variable_spectrum=variable_spectrum, spectrum_filter=spectrum_filter,
                                 reference_spectrum=reference_spectrum and not is_background,
                                 detailed_svl=detailed_svl, device_serial=device_serial)
            try:
                data = simulate(config, rng, None if is_background else spectrum)
            except ValueError as error:
                raise ValueError(f"{point_name} · {label} · {error}") from error
            filename = f"SIMULASYON_{record_name}_{suffix}.SVL"
            file_stem = record_name
            if record_count > 1:
                sequence = f"{len(rows) + 1:0{sequence_width}d}"
                file_stem = record_name[:8 - len(sequence)] + sequence
            rows.append((label, filename, config, data, svl(config, data, file_stem)))
        next_start = config.start + duration
    return title, rows, site_notes


def show_results(st):
    result = st.session_state.get("results")
    if not result:
        st.info("Henüz simülasyon kaydı oluşturulmadı.")
        return
    title, rows, site_notes = result
    section(st, "Simülasyon sonuçları")
    st.write(title)
    workspace = st.session_state.noise_workspace
    saved_inputs = (workspace.get("run") or {}).get("inputs", {})
    saved_maximum = saved_inputs.get("maximum_peaks")
    draft_maximum = workspace["draft"].get("maximum_peaks", saved_maximum)
    applied_filter = rows[0][2].spectrum_filter
    if workspace["draft"].get("spectrum_filter", applied_filter) != applied_filter:
        st.info("Spektrum filtresi değişti. Ölçüm oluştur ekranında yeni sonuçları oluştur.")
        return
    if draft_maximum != saved_maximum:
        st.info("Pik sınırı değişti. Ölçüm oluştur ekranında yeni sonuçları oluştur.")
        return
    if saved_maximum is not None:
        applied_maximum = min(saved_maximum, (rows[0][2].seconds - 1) // 2)
        st.caption(f"Bu çalışmada kaynak açık kayıt başına seçim aralığı 0–{applied_maximum} yükseliş.")
    if site_notes:
        st.caption(site_notes)
    if len(rows) > 1:
        with st.expander("Ölçüm zaman çizelgesi"):
            st.dataframe([
                {"Kayıt": label, "Başlangıç": config.start.strftime("%d.%m.%Y %H:%M:%S"),
                 "Bitiş": (config.start + timedelta(seconds=config.seconds)).strftime("%d.%m.%Y %H:%M:%S"),
                 "LAeq (dB(A))": average_centi(data.a) / 100,
                 "LCeq (dB(C))": average_centi(data.c) / 100,
                 "Grafikte işaretli yükseliş": len(data.peak_starts),
                 "LCeq − LAeq (dB)": (average_centi(data.c) - average_centi(data.a)) / 100}
                for label, _, config, data, _ in rows
            ], hide_index=True)
    if any(config.start.second % 2 for _, _, config, _, _ in rows):
        st.caption("Çizelge tam başlangıç saatlerini gösterir. SVL başlığı tek saniyeleri önceki çift saniyeye yazar.")
    selection_key = "noise_field_selected_record"
    selected = st.session_state.get(selection_key, workspace["draft"].get("selected_record", 0))
    if not isinstance(selected, int) or selected not in range(len(rows)):
        selected = 0
    st.session_state[selection_key] = selected

    def save_selected_record():
        workspace["draft"]["selected_record"] = st.session_state[selection_key]

    selection = st.selectbox(
        "Kayıt", options=range(len(rows)), key=selection_key,
        format_func=lambda index: rows[index][0], on_change=save_selected_record,
    )
    workspace["draft"]["selected_record"] = selection
    label, filename, config, data, file_bytes = rows[selection]
    chart_key = f"{(workspace.get('run') or {}).get('seed', 'restored')}_{selection}"
    st.caption(f"Gösterilen kayıt · {label}")
    cols = st.columns(2)
    cols[0].metric("LAeq · dB(A)", f"{average_centi(data.a) / 100:.2f}".replace(".", ","))
    cols[1].metric("LCeq · dB(C)", f"{average_centi(data.c) / 100:.2f}".replace(".", ","))
    st.caption(f"{config.start:%d.%m.%Y %H:%M:%S} · {config.seconds // 60} dk · "
               f"{len(data.a)} kayıt · {len(data.peak_starts)} işaretli yükseliş")
    if saved_maximum is not None:
        st.metric("Bu kayıttaki pik / girilen üst sınır", f"{len(data.peak_starts)} / {saved_maximum}")
    show_noise_chart(st, data, key=f"noise_history_{chart_key}")
    with st.expander(f"Ortalama ⅓ oktav spektrumu · {config.spectrum_filter} filtresi"):
        spectrum = {
            "Hz": [f"{frequency:g}" for frequency in FREQUENCIES],
            "Leq (dB)": [level / 100 for level in average_bands(data)],
        }
        st.vega_lite_chart(
            [{"Hz": frequency, "Leq": level} for frequency, level in zip(spectrum["Hz"], spectrum["Leq (dB)"])],
            {
                "mark": {"type": "bar", "tooltip": True},
                "encoding": {
                    "x": {"field": "Hz", "type": "ordinal", "sort": spectrum["Hz"]},
                    "y": {"field": "Leq", "type": "quantitative", "title": "Leq (dB)"},
                },
            },
            width="stretch", key=f"noise_spectrum_{chart_key}",
        )
        st.dataframe(spectrum, hide_index=True)
    st.download_button("Simülasyon SVL dosyasını indir", file_bytes, filename,
                       "application/octet-stream", use_container_width=True, on_click="ignore")
    if len(rows) > 1:
        archive = BytesIO()
        with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
            for _, member, _, _, payload in rows:
                bundle.writestr(member, payload)
        st.download_button("Tüm simülasyon kayıtlarını indir", archive.getvalue(),
                           "SIMULASYON_SVL.zip", "application/zip",
                           use_container_width=True, on_click="ignore")
    st.caption("Simülasyon verisi · Eğitim ve yazılım testi")


def show_noise_chart(st, data, key=None):
    peak_starts = set(data.peak_starts)
    series = [
        {"Saniye": index + 1, "LAeq": a / 100, "LCeq": c / 100, "yukselis": index in peak_starts}
        for index, (a, c) in enumerate(zip(data.a, data.c))
    ]
    st.vega_lite_chart(
        series,
        {
            "height": 300,
            "encoding": {"x": {"field": "Saniye", "type": "quantitative"}},
            "layer": [
                {
                    "transform": [{"fold": ["LAeq", "LCeq"], "as": ["Profil", "Seviye"]}],
                    "mark": {"type": "line", "tooltip": True},
                    "encoding": {
                        "y": {"field": "Seviye", "type": "quantitative", "title": "Seviye (dB)",
                              "scale": {"zero": False}},
                        "color": {"field": "Profil", "type": "nominal", "legend": {"orient": "bottom"}},
                    },
                },
                {
                    "transform": [{"filter": "datum.yukselis"}],
                    "mark": {"type": "point", "shape": "triangle-up", "filled": True,
                             "color": "#f59e0b", "size": 95},
                    "encoding": {
                        "y": {"field": "LAeq", "type": "quantitative", "title": "Seviye (dB)",
                              "scale": {"zero": False}},
                        "tooltip": [
                            {"field": "Saniye", "type": "quantitative", "title": "Yükseliş başlangıcı (sn)"},
                            {"field": "LAeq", "type": "quantitative", "title": "LAeq (dB(A))", "format": ".2f"},
                        ],
                    },
                },
            ],
        },
        width="stretch", key=key,
    )
    if peak_starts:
        st.caption("Sarı üçgenler simülasyondaki yükselişlerin başladığı saniyeleri gösterir.")


WORKSPACE_STORAGE = """
const storageKey = "gurultu-saha-asistani.workspace.v1";
const lastWrite = new WeakMap();
let failed = false;

export default function ({data, parentElement, setStateValue}) {
  if (!data) return;
  try {
    if (data.read) {
      setStateValue("loaded", localStorage.getItem(storageKey) ?? "");
    } else if (lastWrite.get(parentElement) !== data.workspace) {
      localStorage.setItem(storageKey, data.workspace);
      lastWrite.set(parentElement, data.workspace);
    }
  } catch {
    if (data.read) setStateValue("loaded", "");
    if (!failed) {
      failed = true;
      setStateValue("failed", true);
    }
  }
}
"""


STYLE = """
<style>
.st-key-noise-storage {display:none;}
.block-container {max-width:1280px;padding-top:2rem;}
.noise-hero {background:linear-gradient(120deg,#102e40,#081b2a);
    color:#edf6fa;padding:26px 24px;margin:0 0 22px;border-radius:24px;}
.noise-hero h1 {color:#edf6fa;margin:0;padding:0;font-size:clamp(25px,6vw,48px);}
.noise-hero p {margin:8px 0 0;color:#b8d5de;}
.noise-section {margin:19px 0 8px;padding:10px 14px;border-left:3px solid #20b8a7;
    border-radius:0 12px 12px 0;background:linear-gradient(100deg,rgba(22,163,155,.14),rgba(22,163,155,.02));}
.noise-section h3 {margin:0;padding:0;font-size:19px;}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#102e40,#081b2a 65%)!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] p {color:#edf6fa!important;}
button[kind="primary"] {background:#078e89!important;border-color:#20bdb1!important;color:white!important;}
[data-testid="stSidebarCollapsedControl"] button::after {content:" Menü";font-weight:650;}
@media(max-width:640px) {
  .block-container {padding:2.8rem 1rem 2rem;}
  .noise-hero {padding:16px;border-radius:18px;}
  .st-key-datetime [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;gap:.5rem;}
  .st-key-datetime [data-testid="stColumn"] {min-width:0;width:50%!important;flex:1 1 50%!important;}
}
</style>
"""

DRAWER = """
<script>
(() => {
  const doc = parent.document;
  if (doc.documentElement.dataset.noiseSwipe) return;
  doc.documentElement.dataset.noiseSwipe = "1";
  let start;
  doc.addEventListener("touchstart", event => {
    start = null;
    if (!parent.matchMedia("(max-width:640px)").matches || event.touches.length !== 1) return;
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    const point = event.touches[0];
    const open = sidebar && sidebar.getAttribute("aria-expanded") === "true";
    if ((!open && point.clientX > 26) || (open && !sidebar.contains(event.target))) return;
    if (event.target.closest("input, select, textarea, button, canvas")) return;
    start = [point.clientX, point.clientY, open];
  }, {passive:true});
  doc.addEventListener("touchcancel", () => { start = null; }, {passive:true});
  doc.addEventListener("touchend", event => {
    if (!start) return;
    const [x, y, open] = start;
    start = null;
    const end = event.changedTouches[0];
    if (!end || Math.abs(end.clientY - y) > 55) return;
    const delta = end.clientX - x;
    if (!open && delta > 60) {
      doc.querySelector('[data-testid="stSidebarCollapsedControl"] button')?.click();
    } else if (open && delta < -60) {
      doc.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click();
    }
  }, {passive:true});
})();
</script>
"""


if __name__ == "__main__":
    run_app()
