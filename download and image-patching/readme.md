<h2 dir="rtl"> آماده‌سازی تصاویر </h2>

<p dir="rtl">
داده‌ی مورد نیاز برای پروژه از دو قسمت متن گزارش و تصاویر مرتبط با آن گزارش، تشکیل شده‌است. برای استفاده و دانلود تصاویرwhole slide image، نیاز است UUID هر کدام از تصاویر را پیدا کنیم. این UUID نشانگر یک رشته‌ی منحصر به فرد برای هر فایل است که با استفاده از آن فایل مورد نظر به صورت یکتا از دیتابیس استفاده شده قابل دانلود است. برای این کارmanifest   کامل دیتاست مورد نظر را از این <a href="https://portal.gdc.cancer.gov/projects/TCGA-BRCA"> آدرس </a> دانلود می‌کنیم که شامل UUID، نام فایل، سایز فایل و موارد دیگر است. فایل‌های انتخابی زیرمجموعه‌ی این فایل‌ها هستند. از روی نام فایل آن‌هایی که فرمت svs دارند و جزو گزارش‌های انتخاب‌شده هستند را فیلتر می‌کنیم. نهایتا یک فایل manifest خواهیم داشت که تمام موارد آن باید دانلود و پردازش شود.</p>



<p dir="rtl">
با توجه به این ‌که تصاویر موردنظر به صورت مستقیم از منبع انتخاب شده قابلیت دانلود شدن را ندارند، نیاز است برنامه‌ی  <a href="https://github.com/NCI-GDC/gdc-client">gdc client</a> که پیشنهاد خود منبع است، نصب شود. این برنامه در دو نسخه‌ی دارای محیط گرافیکی و بدون محیط گرافیکی ارائه می‌شود که با توجه به فرآیند انجام کار که در colab اجرا می‌شود، نسخه‌ی بدون محیط گرافیکی، نصب شده‌است. این برنامه UUID ذکر شده در مرحله‌ی قبل را به عنوان ورودی گرفته و فایل مورد نظر را دانلود و ذخیره می‌کند. در colab فایل دانلود شده در root پروژه ذخیره می‌شود. با توجه به فایل manifest ساخته شده، فایل‌ها را یکی یکی دانلود می‌کنیم. تصاویر whole slide image دانلود شده حجمی بین چند ده مگابایت تا چند گیگابایت دارند که در ابعاد تصویر نیز متفاوت هستند. مجموع حجم فایل‌های دانلود شده حدودا ۶۰ گیگابایت است.
</p>

<p dir="rtl">
یک نمونه از تصویر WSI دانلود شده -برای بافت سینه- به صورت ریسایز شده در پایین صفحه آمده‌است. تصاویر دانلود شده در نهایت برای استفاده در مدل نهایی باید به فرمت jpg یا png تبدیل شوند. 
</p>

<p dir="rtl">
برای خواندن تصاویر با فرمت svs، از کتابخانه‌ی <a href="https://github.com/openslide/openslide-python">openslide </a>  استفاده کرده‌ایم. این کتابخانه تابعی را معرفی می‌کند که با فراخوانی آن می‌توانیم بخشی از تصویر را در پایتون برای پردازش کردن، بارگذاری کنیم. امکان بارگذاری کامل تصویر با توجه به حجم زیاد آن و محدودیت حافظه‌ی colab، وجود ندارد. 
</p>


<p dir="rtl">
با توجه به هدف پروژه، نیاز است تصاویر دانلود شده به صورت پچ‌هایی با سایز کمتر تبدیل شوند. برای این کار روی عکس در جهت سطری و در جهت ستونی با گام‌های ۱۰۲۴ پیکسلی حرکت می‌کنیم و ناحیه انتخاب شده را با استفاده از openslide می‌خوانیم. با این کار عکس مورد نظر به پچ‌هایی با سایز ۱۰۲۴ در ۱۰۲۴ پیکسل تبدیل می‌شود. از آنجا که همه‌ی این پچ‌ها حاوی شی‌ مورد نظر نیستند و اطلاعات مهمی در خود ندارند و عمدتا سفید رنگ هستند، نیاز است از مجموعه‌ی پچ‌های انتخاب شده حذف شوند. برای این‌کار از الگوریتم آماده‌ی otsu استفاده می‌کنیم. این الگوریتم تصویر را به دو دسته‌ی پس‌زمینه و پیش‌زمینه، تقسیم می‌کند. اگر درصد زیادی از تصویر مورد نظر به پس زمینه، که سفید است، تعلق داشته باشد، آن را حذف می‌کنیم. این کار را با تعریف یک threshold ثابت انجام می‌دهیم. مشکل این روش در آن است که برای تصاویری که اکثر پیکسل‌های آن سفید رنگ است، کل عکس را به عنوان پیش‌زمینه در نظر می‌گیرد. برای حل این موضوع یک نسبت دیگری تعریف می‌شود که میزان پیکسل‌های سفید رنگ به کل پیکسل‌های تصویر را حساب می‌کند، اگر پیکسلی در حالت gray scale، اگر مقدار بزرگتر از ۲۲۵ داشته باشد پیکسل سفید حساب می‌شود. اگر برای پچی این نسبت از مقدار مشخصی بالاتر باشد، آن پچ انتخاب نخواهد شد. در نهایت پچ‌هایی باقی خواهد ماند که حاوی اطلاعات مهم هستند. این کار را بار دیگر با گام ۲۰۴۸ و ۳۰۷۲ پیکسل انجام می‌دهیم. 
</p>

<p dir="rtl">
خروجی این بخش برای هر تصویر شامل سه سری پچ‌های با سایزهای ۱۰۲۴، ۲۰۴۸ و ۳۰۷۲ ‌است که در نهایت برای نرمال کردن به سایز ۲۹۹ در ۲۹۹ پیکسل ریسایز شده‌اند.
</p>

<hr>
<h3 dir="rtl">مثال</h3>
<p dir="rtl"> تصویر کامل </p>
<img src="../data-example/main.png" />

<br/>
<p dir="rtl">نمونه‌ای از پچ‌های ساخته‌شده از تصویر بالا</p>
<img src="../data-example/6644_12788.png" />
<br />
<img src="../data-example/6644_15860.png" />