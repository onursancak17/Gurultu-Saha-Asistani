"""SVAN 971 için etiketli eğitim ve yazılım testi simülasyonu."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from html import escape
from io import BytesIO
from math import cos, isfinite, log10, log2, sin
import re
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
    event_gain: float = 8.0
    delay: int = 3
    high_range: bool = False


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


def spectrum_offsets(c_minus_a):
    # Two broad, synthetic spectral shapes. Mix in the linear energy domain.
    def shape(center, width):
        bins = [energy(-3 * (log2(freq / center) / width) ** 2) for freq in FREQUENCIES]
        a_energy = sum(value * energy(weight) for value, weight in zip(bins, A_WEIGHT))
        normalized = [value / a_energy for value in bins]
        c_energy = sum(value * energy(weight) for value, weight in zip(normalized, C_WEIGHT))
        return normalized, c_energy

    high, high_c = shape(1800, 0.95)
    low, low_c = shape(125, 1.1)
    ratio = energy(c_minus_a)
    if not high_c <= ratio <= low_c:
        raise ValueError(
            f"Bu spektrum modeli için LCeq − LAeq farkı "
            f"{10 * log10(high_c):.2f} ile {10 * log10(low_c):.2f} dB arasında olmalı."
        )
    portion = (ratio - high_c) / (low_c - high_c)
    bins = [hi * (1 - portion) + lo * portion for hi, lo in zip(high, low)]
    return tuple(round(1000 * log10(value)) for value in bins), round(1000 * log10(sum(bins)))


def simulate(config):
    if not 1 <= config.seconds <= 3600:
        raise ValueError("Bir kayıt 1 ile 3600 saniye arasında olmalı.")
    if not isfinite(config.a) or not isfinite(config.c):
        raise ValueError("LAeq ve LCeq sonlu sayı olmalı.")
    if config.event_count < 0 or config.event_seconds < 1 or not 0 < config.event_gain <= 40:
        raise ValueError("Pik sayısını, süresini ve seviyesini kontrol et.")
    occupied = config.event_count * config.event_seconds
    free = config.seconds - occupied
    if config.event_count and free < config.event_count + 1:
        raise ValueError("Pikler arasında ve kaydın iki ucunda en az birer normal saniye bırakılmalı.")

    # Distribute complete events with a normal interval between each event.
    events = [False] * config.seconds
    for index in range(config.event_count):
        start = (index + 1) * free // (config.event_count + 1) + index * config.event_seconds
        events[start : start + config.event_seconds] = [True] * config.event_seconds
    relative = [
        0.31 * sin(index / 5.7) + 0.19 * cos(index / 11.1)
        + (config.event_gain if active else 0)
        for index, active in enumerate(events)
    ]
    normalization = 10 * log10(sum(map(energy, relative)) / config.seconds)
    a = tuple(round(100 * (config.a + level - normalization)) for level in relative)
    c = tuple(level + round(100 * (config.c - config.a)) for level in a)
    minimum, upper_peak = (30, 140) if config.high_range else (25, 123)
    if min(a) < minimum * 100 or max(max(a), max(c)) >= upper_peak * 100:
        raise ValueError("Senaryodaki seviyeler seçili cihaz aralığına sığmıyor. Seviyeleri veya aralığı düzenle.")
    offsets, z = spectrum_offsets(config.c - config.a)
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
    html(DRAWER, height=0)
    with st.sidebar:
        st.markdown("## Gürültü Saha Asistanı")
        page = st.radio("Menü", ("Ana ekran", "Ölçüm oluştur", "Sonuçlar"), label_visibility="collapsed")
        st.caption("SVAN 971 · Simülasyon")

    st.markdown(
        '<header class="noise-hero"><h1>Gürültü Saha Asistanı</h1>'
        '<p>Svantek SVAN 971 · Simülasyon</p></header>',
        unsafe_allow_html=True,
    )
    if page == "Sonuçlar":
        show_results(st)
        return
    if page == "Ana ekran" and not st.session_state.get("device_selected"):
        with st.container(border=True):
            st.subheader("SVAN 971")
            st.caption("Svantek · Çevresel gürültü")
            if st.button("SVAN 971’i seç", type="primary", use_container_width=True):
                st.session_state.device_selected = True
                st.rerun()
        return

    inputs = measurement_form(st)
    if st.button("Ölçüm Yap", type="primary", use_container_width=True):
        try:
            st.session_state.results = build_results(**inputs)
        except ValueError as error:
            st.error(str(error))
            return
    if st.session_state.get("results"):
        show_results(st)


def section(st, label):
    st.markdown(f'<div class="noise-section"><h3>{escape(label)}</h3></div>', unsafe_allow_html=True)


def measurement_form(st):
    section(st, "Ölçüm metodu")
    period = st.selectbox("TS ISO 1996-1/2 · Çevresel gürültü", PERIODS)
    section(st, "Tesis ve ölçüm noktası")
    plant = st.text_input("Tesis adı")
    point = st.text_input("Ölçüm noktası")
    setting = st.selectbox("Ortam", ("Dış ortam", "İç ortam"))
    mode = st.text_input("İşletme modu", placeholder="Örn. tesis tam yükte")
    with st.expander("Saha bilgileri"):
        placement = st.selectbox("Mikrofon konumu", ("Belirtilmedi", "Serbest alan", "Cephe önü", "Yüzeye monteli", "Oda içi"))
        height = st.number_input("Mikrofon yüksekliği (m)", min_value=0.0, value=None, step=0.1)
        distance = st.number_input("Kaynağa yatay uzaklık (m)", min_value=0.0, value=None, step=0.5)
        weather = st.text_input("Hava koşulları ve saha notu", placeholder="Rüzgâr hızı/yönü, sıcaklık, nem ve yağış")
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
        day = date_col.date_input("Başlangıç tarihi", value=now.date(), format="DD.MM.YYYY")
        clock = time_col.time_input("Başlangıç saati", value=now.time().replace(second=0, microsecond=0), step=60)
    default_minutes = (5, 10, 30)[PERIODS.index(period)]
    period_min = st.number_input("Kaynağın tam çalışma döngüsü (dk, biliniyorsa)", 0, 60, 0)
    recommended = max(default_minutes, period_min)
    manual = st.toggle("Ölçüm süresini manuel belirle")
    if manual:
        minutes = st.number_input("Ölçüm süresi (dk)", 1, 60, recommended, key=f"duration-{recommended}")
    else:
        minutes = recommended
        st.caption(f"Örnek kayıt süresi {minutes} dk")
    if minutes < recommended:
        st.warning(f"Seçilen saha planında önerilen süre {recommended} dk. Simülasyon oluşturulabilir.")
    repeats = st.number_input("Cihaz tekrar sayısı", 1, 10, 1)

    section(st, "Kayıt ve ses seviyesi")
    name = st.text_input("SVL kayıt adı", value="ORCALI4", max_chars=28)
    a = st.number_input("Kaynak açık · Hedef toplam LAeq [dB(A)]", 25.0, 139.0, 60.0, 0.1)
    character = st.radio("Ses karakteri", ("Kararlı", "Pikli"), horizontal=True)
    count = 0
    length = 2
    gain = 8.0
    if character == "Pikli":
        count = st.number_input("Pik sayısı", 1, 200, 3)
        length = st.number_input("Her pikin süresi (sn)", 1, 120, 2)
        gain = st.number_input("Pikin normal seviyeden farkı (dB)", 1.0, 40.0, 8.0, 0.5)

    with st.expander("Arka plan ve cihaz ayarları"):
        manual_c = st.checkbox("Hedef LCeq’yi manuel belirle")
        c = st.number_input("Hedef LCeq [dB(C)]", -10.0, 145.0, min(145.0, a + 7.0), 0.1) if manual_c else a + 7.0
        has_background = st.checkbox("Kaynak kapalıyken arka plan kaydı ekle")
        background = st.number_input("Arka plan hedef LAeq [dB(A)]", 25.0, 139.0, 45.0, 0.1) if has_background else None
        high_range = st.selectbox("Cihaz aralığı", ("Düşük", "Yüksek")) == "Yüksek"
        delay = st.number_input("Başlangıç gecikmesi (sn)", 0, 60, 3,
                                help="Cihaz ayarı olarak kaydedilir. Formdaki başlangıç, kaydın ilk zamanıdır.")
        st.caption("Kayıt adımı 1 sn · ⅓ oktav · Başlangıç senkronizasyonu kapalı")
    return dict(
        plant=plant, point=point, setting=setting, mode=mode, day=day, clock=clock,
        site_notes=site_notes,
        minutes=minutes, repeats=repeats, name=name, a=a, c=c, count=count,
        length=length, gain=gain, background=background, high_range=high_range, delay=delay,
    )


def build_results(plant, point, setting, mode, day, clock, minutes, repeats, name,
                  a, c, count, length, gain, background, high_range, delay, site_notes=""):
    if not plant.strip() or not point.strip() or not slug(name).strip("_-"):
        raise ValueError("Tesis adı, ölçüm noktası ve SVL kayıt adı gerekli.")
    title = f"{plant.strip()} / {point.strip()}"
    metadata = f"{title} | {setting} | {mode.strip()}"
    start = datetime.combine(day, clock)
    start -= timedelta(seconds=start.second % 2, microseconds=start.microsecond)
    rows = []
    for index in range(repeats + (background is not None)):
        is_background = index == repeats
        label = "Arka plan" if is_background else f"Kaynak açık · Tekrar {index + 1}"
        suffix = "FON" if is_background else f"{index + 1:02d}"
        device_id = slug(name)[:max(1, 5 - len(suffix))] + suffix
        level_a = background if is_background else a
        level_c = background + 7.0 if is_background else c
        config = Measurement(device_id, start + timedelta(minutes=minutes * index),
                             minutes * 60, level_a, level_c, 0 if is_background else count,
                             length, gain, delay, high_range)
        data = simulate(config)
        filename = f"SIMULASYON_{slug(name)}_{suffix}.SVL"
        rows.append((label, filename, config, data, svl(config, data, metadata + " | " + label)))
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
    selection = st.selectbox("Kayıt", range(len(rows)), format_func=lambda index: rows[index][0])
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


STYLE = """
<style>
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
