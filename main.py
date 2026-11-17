"""SVAN 971 için etiketli eğitim ve yazılım testi simülasyonu."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from zoneinfo import ZoneInfo
from html import escape
from io import BytesIO
import json
from math import ceil, floor, isfinite, log10, log2
from random import Random
import re
from secrets import token_hex
from struct import pack
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
PERIODS = ("Türkiye Ek-2 – 5 dk referans", "ISO 1996-2 – elverişli veya geometri koşulu", "ISO 1996-2 – diğer koşullar")
DEFAULT_POINT_VARIATION_DB = 5.0
UNAVAILABLE = 0xD000


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


@dataclass(frozen=True)
class Simulation:
    a: tuple[int, ...]
    c: tuple[int, ...]
    spectrum_offsets: tuple[int, ...]
    z_offset: int


def energy(level):
    return 10 ** (level / 10)


def average_centi(series):
    return round(1000 * log10(sum(energy(level / 100) for level in series) / len(series)))


def weighted_energy(bins, weights):
    return sum(value * energy(weight) for value, weight in zip(bins, weights))


def spectral_shape(center, width):
    bins = [energy(-3 * (log2(freq / center) / width) ** 2) for freq in FREQUENCIES]
    a_energy = weighted_energy(bins, A_WEIGHT)
    normalized = [value / a_energy for value in bins]
    return normalized, weighted_energy(normalized, C_WEIGHT)


def spectral_endpoints(extended=False):
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


def spectrum_offsets(bins):
    return tuple(round(1000 * log10(value)) for value in bins), round(1000 * log10(sum(bins)))


def normal_profile(seconds, rng):
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


def simulate(config, rng=None, spectrum=None):
    if not 1 <= config.seconds <= 3600:
        raise ValueError("Bir kayıt 1 ile 3600 saniye arasında olmalı.")
    if not isfinite(config.a) or not isfinite(config.c):
        raise ValueError("LAeq ve LCeq sonlu sayı olmalı.")
    minimum_gain = config.event_min_gain
    if minimum_gain is None:
        minimum_gain = min(10.0, config.event_gain)
    if config.event_count < 0 or config.event_seconds < 1 or not 0 < minimum_gain <= config.event_gain <= 40:
        raise ValueError("Pik sayısını, süresini ve dB aralığını kontrol et. Alt sınır üst sınırı aşamaz.")
    occupied = config.event_count * config.event_seconds
    free = config.seconds - occupied
    if config.event_count and free < config.event_count + 1:
        raise ValueError("Pikler arasında ve kaydın iki ucunda en az birer normal saniye bırakılmalı.")

    rng = rng if rng is not None else Random()
    relative = normal_profile(config.seconds, rng)
    if config.event_count:
        add_peak_events(relative, config.event_count, config.event_seconds, minimum_gain, config.event_gain, rng)
    normalization = 10 * log10(sum(map(energy, relative)) / config.seconds)
    a = tuple(round(100 * (config.a + level - normalization)) for level in relative)
    if spectrum is None:
        c_minus_a = config.c - config.a
        spectrum = spectral_mix(c_minus_a, spectral_endpoints())
    else:
        c_minus_a = 10 * log10(weighted_energy(spectrum, C_WEIGHT) / weighted_energy(spectrum, A_WEIGHT))
    c = tuple(level + round(100 * c_minus_a) for level in a)
    minimum, upper_peak = (30, 140) if config.high_range else (25, 123)
    if min(a) < minimum * 100 or max(max(a), max(c)) >= upper_peak * 100:
        raise ValueError("Senaryodaki seviyeler seçili cihaz aralığına sığmıyor. Seviyeleri veya aralığı düzenle.")
    offsets, z = spectrum_offsets(spectrum)
    return Simulation(a, c, offsets, z)


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


def profile_settings():
    return [0x1405, 0x0307, 0x0606, 1, 2, 8, 2, 0, 0x0606, 1, 3, 8, 3, 0, 0x0606, 0, 2, 8, 2, 0]


def svl(config, data, title):
    """Write the SVAN 971 1/3-octave logger format from App. B of the manual."""
    start_date, start_time = encoded_time(config.start)
    device_filename = ("SIM" + slug(config.name).replace("-", "").replace("_", "")[:5]).encode("ascii")
    ascii_title = normalize("NFKD", title.replace("ı", "i").replace("İ", "I"))
    description = ("SIMULASYON | EGITIM / YAZILIM TESTI | " + ascii_title).encode("ascii", errors="ignore")[:126]
    # Match the reference format's fixed title block and include a NUL terminator.
    description = description.ljust(130, b"\0")

    pc_header = b"SvanPC" + words([26, 32, 71] + [0] * 10)
    file_header = words([0x0E01]) + device_filename.ljust(8, b" ") + words(
        [0x0300, start_date, start_time] + [0] * 6
    )
    # Zero serial number and report no field calibration on synthetic data.
    unit = words([0x0B02, 0, 971, 114, 0, 1, 1, 114, 114, 1, 0])
    calibration = words([0x0B47, 0, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 9400, 0xFFFF])
    user_text = words([0x4203]) + description
    general = [
        0x4004, start_date, start_time, 3, 2, 2 if config.high_range else 1, 0, 1, 1, 3,
        config.delay, config.seconds, 0, 2, 0, 1, 8, 480, 0, 1, 0, 0, 800, 0, 3,
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
    samples = bytearray()
    for a, c in zip(data.a, data.c):
        bands = [max(0, a + offset) for offset in data.spectrum_offsets]
        samples += words([0, a, c, a, *bands, a, c, a + data.z_offset])

    main_results = [0x4407, 0x0307]
    for profile, series in enumerate((data.a, data.c, data.a)):
        leq = average_centi(series)
        main_results.extend([
            0x1608, config.seconds if profile == 0 else 0, 0,
            UNAVAILABLE, round(leq + 1000 * log10(config.seconds)),
            UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, leq, *([0] * 11),
            UNAVAILABLE, UNAVAILABLE,
        ])
    spectrum = [0x2710, 0x0101, 2000, 31, 3]
    spectrum += [
        average_centi([max(0, a + offset) for a in data.a])
        for offset in data.spectrum_offsets
    ]
    spectrum += [average_centi(data.a), average_centi(data.c), average_centi(data.a) + data.z_offset]
    summary = record([0x0359, 2, 0]) + record(main_results) + record(spectrum)
    logger_size = len(samples) + len(summary)
    logger = words([0x0E0F, 1, 0, 2000, 31, 3]) + pack(
        "<IIII", logger_size, config.seconds, config.seconds, 0
    )
    return b"".join((
        pc_header, file_header, unit, calibration, user_text, words(general), words(triggers),
        words(profile_settings()), words(display), logger, samples, summary,
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
    if st.button("Ölçüm Yap", type="primary", use_container_width=True):
        try:
            seed = token_hex(16)
            st.session_state.results = build_results(**inputs, seed=seed)
            st.session_state.noise_workspace["run"] = {"inputs": inputs, "seed": seed}
            draft.pop("selected_record", None)
            st.session_state.pop("noise_field_selected_record", None)
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
    with st.container(key="datetime"):
        date_col, time_col = st.columns(2)
        day = field(date_col.date_input, "day", "Başlangıç tarihi", now.date(), format="DD.MM.YYYY")
        clock = field(time_col.time_input, "clock", "Başlangıç saati", now.time().replace(second=0, microsecond=0), step=60)
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
    character = field(st.radio, "character", "Ses karakteri", "Kararlı", options=("Kararlı", "Pikli"), horizontal=True)
    count = 0
    length = 2
    min_gain, gain = 10.0, 25.0
    if character == "Pikli":
        count = field(st.number_input, "peak_count", "Pik sayısı", 3, min_value=1, max_value=200)
        length = field(st.number_input, "peak_length", "Her pikin süresi (sn)", 2, min_value=1, max_value=120)
        low_col, high_col = st.columns(2)
        min_gain = field(low_col.number_input, "min_gain", "En düşük pik artışı (dB)", 10.0,
                         min_value=1.0, max_value=40.0, step=0.5)
        gain = field(high_col.number_input, "gain", "En yüksek pik artışı (dB)", 25.0,
                     min_value=1.0, max_value=40.0, step=0.5)
        st.caption("Artışlar o andaki normal seviyeye eklenir. Pikler dâhil tüm kayıt hedef LAeq’ye ayarlanır.")

    with st.expander("Arka plan ve cihaz ayarları"):
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
        st.caption("Kayıt adımı 1 sn · ⅓ oktav · Başlangıç senkronizasyonu kapalı")
    return dict(
        plant=plant, point=point, setting=setting, mode=mode, day=day, clock=clock,
        site_notes=site_notes,
        minutes=minutes, repeats=repeats, name=name, a=a, c=c, count=count,
        length=length, gain=gain, background=background, high_range=high_range, delay=delay,
        min_gain=min_gain, gap_seconds=gap_seconds, following_points=following_points,
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
                  a_bounds=None, c_bounds=None, manual_targets=()):
    record_name = slug(name)
    if not plant.strip() or not point.strip() or not record_name.strip("_-"):
        raise ValueError("Tesis adı, ölçüm noktası ve SVL kayıt adı gerekli.")
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
    endpoints = spectral_endpoints(extended=True) if a_bounds is not None else None
    plant = plant.strip()
    title = f"{plant} · {len(points)} ölçüm noktası" if multi_point else f"{plant} / {point.strip()}"
    next_start = datetime.combine(day, clock)
    duration = timedelta(minutes=minutes)
    record_interval = duration + timedelta(seconds=gap_seconds)
    rows = []
    rng = Random(seed)
    for number, (point_name, transition) in enumerate(points, start=1):
        next_start += timedelta(seconds=transition)
        metadata = f"{plant} / {point_name} | {setting} | {mode.strip()}"
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
            config = Measurement(device_id, next_start + record_interval * index,
                                 minutes * 60, level_a, level_c, 0 if is_background else count,
                                 length, gain, delay, high_range, event_min_gain=min_gain)
            try:
                data = simulate(config, rng, None if is_background else spectrum)
            except ValueError as error:
                raise ValueError(f"{point_name} · {label} · {error}") from error
            filename = f"SIMULASYON_{record_name}_{suffix}.SVL"
            rows.append((label, filename, config, data, svl(config, data, metadata + " | " + label)))
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
    if site_notes:
        st.caption(site_notes)
    if len(rows) > 1:
        with st.expander("Ölçüm zaman çizelgesi"):
            st.dataframe([
                {"Kayıt": label, "Başlangıç": config.start.strftime("%d.%m.%Y %H:%M:%S"),
                 "Bitiş": (config.start + timedelta(seconds=config.seconds)).strftime("%d.%m.%Y %H:%M:%S"),
                 "LAeq (dB(A))": average_centi(data.a) / 100,
                 "LCeq (dB(C))": average_centi(data.c) / 100,
                 "LCeq − LAeq (dB)": (average_centi(data.c) - average_centi(data.a)) / 100}
                for label, _, config, data, _ in rows
            ], hide_index=True)
    if any(config.start.second % 2 for _, _, config, _, _ in rows):
        st.caption("Çizelge tam başlangıç saatlerini gösterir. SVL başlığı tek saniyeleri önceki çift saniyeye yazar.")
    selection = work_field(st, st.selectbox, "selected_record", "Kayıt", 0,
                           options=range(len(rows)), format_func=lambda index: rows[index][0])
    label, filename, config, data, file_bytes = rows[selection]
    cols = st.columns(2)
    cols[0].metric("LAeq · dB(A)", f"{average_centi(data.a) / 100:.2f}".replace(".", ","))
    cols[1].metric("LCeq · dB(C)", f"{average_centi(data.c) / 100:.2f}".replace(".", ","))
    st.caption(f"{config.start:%d.%m.%Y %H:%M:%S} · {config.seconds // 60} dk · "
               f"{len(data.a)} kayıt · {config.event_count} pik olayı")
    st.line_chart(
        {"Saniye": list(range(1, len(data.a) + 1)),
         "LAeq · dB(A)": [a / 100 for a in data.a],
         "LCeq · dB(C)": [c / 100 for c in data.c]},
        x="Saniye", y=["LAeq · dB(A)", "LCeq · dB(C)"], y_label="Seviye (dB)",
    )
    with st.expander("Ortalama ⅓ oktav spektrumu"):
        spectrum = {
            "Hz": [f"{frequency:g}" for frequency in FREQUENCIES],
            "Leq (dB)": [
                average_centi([max(0, a + offset) for a in data.a]) / 100
                for offset in data.spectrum_offsets
            ],
        }
        st.bar_chart(spectrum, x="Hz", y="Leq (dB)", sort=False)
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
