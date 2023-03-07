# CLIPath
**CLIP for Whole Slide Image & Report**

<b>مدل یادگیری ماشین کلیپ برای آموزش توام متن گزارش و تصویر WSI</b>

<hr>

<div style="direction:rtl;">

<figure>
  <p align="center">
    <img src="documentation/pipline_flow.png" width="100%" style="display: block; margin-left: auto; margin-right: auto;">
  </p>
  <p align="center">تصویر 1 - پایپلاین پردازشی</p>
</figure>
<br>

پایپلاین پردازشی زمان آزمایش به این صورت می‌باشد که دو پایپلاین موازی فعالیت می‌کنند که به ترتیب عبارت اند از پایپلاین پردازشی تصویر و پایپلاین پردازشی متن. (تصویر ۱)
در ابتدا می‌بایست یک تصویر WSI انتخاب شده و گزارش مربوط به آن نیز مشخص شود. سپس تصویر و گزارش به پایپلاین ها وارد می‌شوند.
<br>
پایپلاین پردازش تصویر ابتدا تصویر WSI را به قطعات کوچکی می‌شکند که حاوی اطلاعات مفید باشند و قطعاتی که اطلاعات مفیدی نداشته باشند حذف می‌شوند. (توضیحات بیشتر در <a href="download and image-patching">بخش قطعه‌بندی تصویر</a>) سپس قطعات وارد بخش نرمال سازی می‌شود که متناسب با یک تصویر مرجع نرمال سازی رنگی برای آن‌ها انجام می‌شود. (توضیحات بیشتر در <a href="color-normalization">بخش نرمال‌سازی تصویر</a>)  پس از آن تصاویر نرمال شده را به مدل DINO می‌دهیم و براساس یک ترشلد بر روی نقشه توجه، تصاویری که نقشه توجه با مقادیر بالاتری داشته باشند به عنوان تصاویر مهم انتخاب می‌شوند. تصاویر انتخاب شده به عنوان ورودی در کنار متن انتخاب شده در پایپلاین متنی به مدل کلیپ داده می‌شوند. (توضیحات بیشتر در <a href="model">بخش مدل</a>)
<br>
پایپلاین پردازش متن ابتدا گزارش مربوط به تصویر WSI را پردازش کرده و بخش Final Diagnosis را به متخصص نشان داده و متخصص بخش ارزشمند آن را در حد ۲ تا ۳ جمله انتخاب می‌کند. (یا خلاصه می‌کند.) و به عنوان متن انتخابی به مدل کلیپ داده می‌شود. (توضیحات بیشتر در <a href="text-preperation">بخش پردازش متن</a>)
<br>
<figure>
  <p align="center">
    <img src="documentation/clip_model.png" style="display: block; margin-left: auto; margin-right: auto; width:70%">
  </p>
  <p align="center">تصویر 2 - مدل کلیپ مورد استفاده</p>
</figure>
<br>
مدل کلیپ (تصویر ۲) استفاده شده نیز به این صورت است که تصاویر انتخاب شده در بخش پایپلاین پردازش تصویر را به عنوان ورودی در کنار متن انتخاب شده از گزارش می‌گیرد و در نهایت تصمیم می‌گیرد کدام متن به کدام تصویر نزدیک است. برای مدل زبانی از <a href="https://arxiv.org/pdf/2205.06885.pdf">pathology BERT</a> استفاده شده است که به صورت خاص روی گزارش‌های پاتولوژی آموزش دیده است و نتایج مناسبی را در این بخش ارائه کرده است.</div>

<br>
<hr>

<div style="direction:rtl;">اعضای گروه:</div>
<div style="direction:rtl;">محمدحسین موثقی‌نیا  <a href="https://github.com/MMovasaghi"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="moh.movasaghinia@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>

<div style="direction:rtl;">رضا عباسی  <a href="https://github.com/abbasiReza"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="reza.abbasi.uni@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>

<div style="direction:rtl;">حسین جعفری‌نیا  <a href="https://github.com/hussein-jafarinia"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="hussein.jafarinia@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>

<div style="direction:rtl;">حسن علیخانی  <a href="https://github.com/psk007"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="alikhani.hassan1997@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>

<div style="direction:rtl;">مهدی شادروی  <a href="https://github.com/MahdiShadrooy"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="mahdishadroy@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>

<div style="direction:rtl;">علی سلمانی  <a href="https://github.com/garamaleki"><img src="documentation/github-logo.jpeg" width="15">  </a><a href="alisalmani200149@gmail.com"><img src="documentation/gmail-logo.webp" width="15"></a></div>