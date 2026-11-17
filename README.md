# Gürültü Saha Asistanı

SVAN 971 ölçüm akışını esas alan Streamlit eğitim ve yazılım testi uygulaması.
Çevresel gürültü formu kararlı ve pikli senaryolar, isteğe bağlı arka plan
kaydı ve simülasyon verisi içeren SVL çıktıları oluşturur.

## Çalıştırma

Python 3.11 veya 3.12 ile:

```sh
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Streamlit Community Cloud için `main.py` ve `requirements.txt` dosyalarını
depo köküne koy ve giriş dosyası olarak `main.py` seç.

## SVL deneme dosyası

`SIMULASYON_ORNEK.SVL` dosyası 300 saniye ve 1 saniye kayıt adımı içerir.
Beklenen değerler LAeq 60,00 dB(A), LCeq 67,00 dB(C) ve üç adet iki saniyelik
yükselmedir. SvanPC++'ta açtıktan sonra kayıt süresini, LAeq ve LCeq
özetini, zaman serisini ve 31 bantlı spektrumu karşılaştır.

Dosya başlığında ve adında simülasyon etiketi bulunur. Örnek gerçek saha
ölçümü veya kalibrasyon bilgisi içermez. LCpeak ve zaman ağırlıklı maksimum
gibi saniyelik LAeq'den çıkarılamayan sonuçlar örnek dosyada doldurulmaz.

Bu ilk sürüm TS ISO 1996-1/2 formunu içerir. ISO 8297 güzergâh ve ses gücü
hesabı sonraki sürüm kapsamındadır. SVL dosya yapısı gerçek örnek kayda ve
üreticinin teknik kılavuzuna göre kontrol edildi. SvanPC++ uygulamasıyla
açılış testi bekliyor.

[SVAN 971 üretici kılavuzu, Ek B](https://svantek.com/wp-content/uploads/2020/08/svan_971_man_en_v.3.1_2020-09-03.pdf)
