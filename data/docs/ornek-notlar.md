# Örnek Bilgi Dosyası

Bu klasördeki `.md`, `.txt`, `.csv`, `.json` dosyaları açılışta otomatik olarak
parçalanır, embedding'e sokulur ve Qdrant'a yazılır. Asistan bunlara
`knowledge_search` aracıyla ve her soruda çalışan otomatik RAG ile erişir.

Dosyayı değiştirip `make rescan` derseniz sadece değişen dosyalar yeniden
işlenir (içerik hash'i tutulur).

## Test için birkaç örnek bilgi

Laboratuvardaki Jetson kartının adı "Toros". Ağ üzerindeki sabit adresi
192.168.1.50 ve odanın anahtarı üçüncü çekmecede duruyor.

Haftalık ekip toplantısı her salı saat 10:30'da yapılıyor.

Kahve makinesinin bakımı ayın ilk cuması yapılır; kartuş değişimi için
depo sorumlusuna haber verilmesi gerekiyor.

---

Bunu silip kendi notlarınızı, el kitaplarınızı veya dökümanlarınızı
koyabilirsiniz.
