
# YOLOv8 Person Tracking and Counting

YOLOv8-Person-Tracking-and-Counting, toplu taşıma, market, mağaza ve cadde gibi alanlardaki insan geçiş verilerini almak için geliştirilmiş bir projedir. Bu proje, nesne takibi ve sayımı için gelişmiş bilgisayarla görme tekniklerini kullanır ve aşağıdaki teknolojileri içerir:

**YOLOv8**: Nesne algılama ve izleme için kullanılır. YOLO (You Only Look Once) modeli, hızlı ve yüksek doğrulukta nesne algılama yeteneklerine sahiptir.

**SORT**: Nesne izleme için kullanılan bir algoritmadır. "Simple Online and Realtime Tracking" (SORT), gerçek zamanlı nesne izleme sağlamak için kullanılır(https://github.com/abewley/sort/tree/master).

**OpenCV**: Görüntü işleme ve bilgisayarla görme kütüphanesi olarak kullanılır. Bu proje, video akışlarından insanları izlemek ve saymak için OpenCV'yi kullanır.


## Kullanım Alanları

Toplu Taşıma: Otobüs, metro ve tren istasyonlarında yolcu yoğunluğunu ve hareketliliğini izlemek.

Market ve Mağazalar: Müşteri trafiğini takip etmek ve yoğunluk verilerini analiz etmek.

Caddeler ve Sokaklar: Yaya trafiğini izlemek ve güvenlik analizleri yapmak.

Bu proje, gerçek zamanlı verileri analiz ederek insan geçiş verilerini toplamak ve anlamlı bilgiler elde etmek için kullanılabilir.
  
## Yazarlar ve Teşekkür

- [@BatuhanTurk](https://github.com/BatuhanTurk) tasarım ve geliştirme için.

  
## Output

![Count People Video ](https://github.com/ulascankutay/YOLOv8-Person-Tracking-and-Counting/blob/main/sonuc%20video/SONUC3.mp4)

  
## Dependencies:

Bu projeyi çalıştırmak için yükleyin

```bash
  $ pip install -r requirements.txt
```

  
## Run :

```bash
  $ git clone https://github.com/ulascankutay/YOLOv8-Person-Tracking-and-Counting.git
```

```bash
  $ python3 thread.py
```
## Lisans

[MIT](https://github.com/ulascankutay/YOLOv8-Person-Tracking-and-Counting/blob/main/MIT.txt)

  