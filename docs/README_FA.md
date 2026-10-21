# راهنمای اجرای نسخهٔ تکمیل‌شده

گزارش دقیق بررسی فصل ۳ و پوشش فصل ۴ در [AUDIT_FA.md](AUDIT_FA.md) و تمام دستورهای اجرایی در [README اصلی](../README.md) آمده است.

ترتیب اجرا:

۱. نصب Python و PyTorch سازگار با GPU و سپس `pip install -e '.[dev,plots,baselines]'`.
۲. دریافت چهار منبع رسمی با `scripts/fetch_upstream.py` و اجرای آزمون‌های عادی و upstream.
۳. تبدیل Foursquare NYC/Tokyo کامل و نسخهٔ pinned Gowalla-CA به CSV استاندارد؛ اجرای audit هر سه.
۴. اجرای سریالی `calibrate` روی GPU مرجع با seed ۱۱؛ این مرحله از کل دادهٔ مجاز آموزش و validation استفاده می‌کند و test را ارزیابی نمی‌کند.
۵. تولید برنامه با `plan --suite full --settings-dir configs/frozen`؛ حفظ تنظیمات و هزینه‌های انتخاب‌شده.
۶. اجرای worker روی GPUهای ۰ و ۱ یک دستگاه با filesystem مشترک؛ هر job روی یک GPU مستقل اجرا می‌شود.
۷. پس از اتمام آموزش و توقف jobهای دیگر، اجرای `worker --gpu 0 --timing`.
۸. تولید `report`، `compare-family` و `diagnostics` و استفاده از خروجی‌های واقعی برای فصل ۴.

داده، checkpoint، cache، تنظیمات frozen و state برنامه را میان قطع‌شدن sessionها نگه دارید. زمان‌بندی برای دو ماشین جدا با Drive همگام‌شونده طراحی نشده است. نسخهٔ جدید checkpointهای نسخهٔ ۰٫۱ را قبول نمی‌کند.

در این بسته نتایج ساختگی مقاله وجود ندارد. نسخه‌های سازگارشدهٔ baselineها، محدودیت‌های پروفایل حافظه و وضعیت آزمون GPU در مستندات صریح‌اند.
