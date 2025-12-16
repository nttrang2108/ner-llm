# LLM Name Entity Recognition (NER)

# 1. Dataset

## 1.1 Raw dataset

Folder structure:

```txt
data/vlps_2018_ner
├── processed
└── raw
    ├── dev
    │   ├── Doi Song
    │   ├── Giai tri
    │   ├── Giao duc
    │   ├── KH - CN
    │   ├── Kinh te
    │   ├── Phap luat
    │   ├── The gioi
    │   ├── The thao
    │   ├── Van hoa
    │   └── Xa hoi
    ├── test
    │   ├── Cong nghe
    │   ├── Cong nghe Thiet bi - Phan cung
    │   ├── Cong nghe - Vien thong
    │   ├── Doi song
    │   ├── Giai tri
    │   ├── Giai tri - Am nhac
    │   ├── Giao duc
    │   ├── Khoa hoc
    │   ├── Kinh te
    │   ├── Kinh te Chung khoan
    │   ├── Kinh te Kinh doanh
    │   ├── Kinh te Tai chinh
    │   ├── Nha dat
    │   ├── Phap luat
    │   ├── The gioi
    │   ├── The thao
    │   ├── The thao Bong da quoc te
    │   ├── The thao Quan vot
    │   ├── Van hoa
    │   ├── Van hoa Du lich
    │   ├── Van hoa Nghe thuat
    │   ├── Xa hoi
    │   ├── Xa hoi Giao thong
    │   ├── Xa hoi Moi truong - Khi hau
    │   └── Xa hoi Thoi su
    └── train
        ├── Doi song
        ├── Giai tri
        ├── Giao duc
        ├── KH - CN
        ├── Kinh te
        ├── Phap luat
        ├── The gioi
        ├── The thao
        ├── Van hoa
        └── Xa hoi
```

Example of a file ".muc":

```python
Người phụ nữ nguy kịch sau phẫu thuật gọt cằm: Chuyển bệnh nhân qua <ENAMEX TYPE="LOCATION">Singapore</ENAMEX> điều trị

Sau thời gian điều trị tại <ENAMEX TYPE="LOCATION">BV Nhân dân 115</ENAMEX>, <ENAMEX TYPE="LOCATION">TP.HCM</ENAMEX>, chị <ENAMEX TYPE="PERSON">Đ.</ENAMEX> đã được rút ống thở và chuyển sang <ENAMEX TYPE="LOCATION">Singapore</ENAMEX> tiếp tục điều trị.

Liên quan đến vụ một người phụ nữ nguy kịch sau phẫu thuật gọt cằm, theo thông tin mới nhất từ <ENAMEX TYPE="ORGANIZATION">Bệnh viện Nhân dân 115</ENAMEX>, <ENAMEX TYPE="LOCATION">TP.HCM</ENAMEX>, sau thời gian điều trị tại BV, chị <ENAMEX TYPE="PERSON">Trần Thị Đ.</ENAMEX> (35 tuổi, ngụ <ENAMEX TYPE="LOCATION">quận Thủ Đức</ENAMEX>) đã rút được ống thở, có đáp ứng được với các kích thích và không còn co giật dù vẫn phải truyền dinh dưỡng qua đường tĩnh mạch.
Thấy tình trạng sức khỏe của chị <ENAMEX TYPE="PERSON">Đ.</ENAMEX> có tiến triển, gia đình chị đã đề xuất nguyện vọng đưa qua <ENAMEX TYPE="LOCATION">Singapore</ENAMEX> chữa trị.
Sau khi tiến hành các thủ tục cần thiết, ngày 22/9 nữ bệnh nhân đã được chuyển từ <ENAMEX TYPE="LOCATION">BV Nhân dân 115</ENAMEX> sang <ENAMEX TYPE="LOCATION">Singapore</ENAMEX> để tiếp tục điều trị. Khi chuyển viện, chị <ENAMEX TYPE="PERSON">Đ.</ENAMEX> không còn phải thở bằng máy.
Như đã thông tin, ngày 17/9, chị <ENAMEX TYPE="PERSON">Đ.</ENAMEX> đến <ENAMEX TYPE="LOCATION">BV thẩm mỹ E.C</ENAMEX> (<ENAMEX TYPE="LOCATION">Quận 10</ENAMEX>, <ENAMEX TYPE="LOCATION">TP.HCM</ENAMEX>) để thực hiện phẫu thuật chỉnh xương hàm trên và dưới.
Ca mổ diễn ra trong 2 giờ đồng hồ. Sau khi mổ xong, bệnh nhân <ENAMEX TYPE="PERSON">Đ.</ENAMEX> tỉnh và được đưa vào phòng hồi sức chừng 20 phút thì bác sĩ gây mê phát hiện có hiện tượng chảy máu phù nề, nhanh chóng rơi vào tình trạng suy hô hấp.
Ngay lập tức, chị <ENAMEX TYPE="PERSON">Đ.</ENAMEX> được <ENAMEX TYPE="ORGANIZATION">BV thẩm mỹ E.C</ENAMEX> mở nội khí quản giúp thở, cầm máu và chuyển đến <ENAMEX TYPE="LOCATION">BV Nhân dân 115</ENAMEX> cấp cứu.
>> XEM THÊM: Thông tin mới nhất vụ người phụ nữ nguy kịch sau phẫu thuật gọt cằm ở <ENAMEX TYPE="LOCATION">Sài Gòn</ENAMEX>
<ENAMEX TYPE="PERSON">Quỳnh Chi</ENAMEX>
```


Entities:

- **`<ENAMEX TYPE="PERSON">...</ENAMEX>`** is a person name.

- **`<ENAMEX TYPE="LOCATION">...</ENAMEX>`** is a location.

- **`<ENAMEX TYPE="ORGANIZATION">...</ENAMEX>`** is an organization name.

## 1.2 Processed dataset

Folder structure:

```txt
data/vlps_2018_ner/processed
├── dev
├── test
└── train
```

Example of a file "dev.json":


```json
[
    {
        // Article 1
    },
    {
        // Article 2
    },
    ...
    {
        "id": 23353874,
        "topic": "Doi song",
        "title": "Người phụ nữ nguy kịch sau phẫu thuật gọt cằm: Chuyển bệnh nhân qua Singapore điều trị",
        "text": "Sau thời gian điều trị tại BV Nhân dân 115, TP.HCM, chị Đ. đã được rút ống thở và chuyển sang Singapore tiếp tục điều trị. Liên quan đến vụ một người phụ nữ nguy kịch sau phẫu thuật gọt cằm, theo thông tin mới nhất từ Bệnh viện Nhân dân 115, TP.HCM, sau thời gian điều trị tại BV, chị Trần Thị Đ. (35 tuổi, ngụ quận Thủ Đức) đã rút được ống thở, có đáp ứng được với các kích thích và không còn co giật dù vẫn phải truyền dinh dưỡng qua đường tĩnh mạch. Thấy tình trạng sức khỏe của chị Đ. có tiến triển, gia đình chị đã đề xuất nguyện vọng đưa qua Singapore chữa trị. Sau khi tiến hành các thủ tục cần thiết, ngày 22/9 nữ bệnh nhân đã được chuyển từ BV Nhân dân 115 sang Singapore để tiếp tục điều trị. Khi chuyển viện, chị Đ. không còn phải thở bằng máy. Như đã thông tin, ngày 17/9, chị Đ. đến BV thẩm mỹ E.C (Quận 10, TP.HCM) để thực hiện phẫu thuật chỉnh xương hàm trên và dưới. Ca mổ diễn ra trong 2 giờ đồng hồ. Sau khi mổ xong, bệnh nhân Đ. tỉnh và được đưa vào phòng hồi sức chừng 20 phút thì bác sĩ gây mê phát hiện có hiện tượng chảy máu phù nề, nhanh chóng rơi vào tình trạng suy hô hấp. Ngay lập tức, chị Đ. được BV thẩm mỹ E.C mở nội khí quản giúp thở, cầm máu và chuyển đến BV Nhân dân 115 cấp cứu. >> XEM THÊM: Thông tin mới nhất vụ người phụ nữ nguy kịch sau phẫu thuật gọt cằm ở Sài Gòn Quỳnh Chi",
        "ground_truth": {
            "person": [
                "Đ.",
                "Trần Thị Đ.",
                "Quỳnh Chi"
            ],
            "organizations": [
                "Bệnh viện Nhân dân 115",
                "BV thẩm mỹ E.C"
            ],
            "address": [
                "BV Nhân dân 115",
                "TP.HCM",
                "Singapore",
                "quận Thủ Đức",
                "BV thẩm mỹ E.C",
                "Quận 10",
                "Sài Gòn"
            ]
        }
    }
]
```

## 1.3 Analysis

Please open [Dataset Analysis](../notebooks/01_Dataset_Analysis.ipynb) notebook for more details.


# 2. LLM Name Entity Recognition (NER)

## 2.1. Prompt Engineering

### 2.1.1. Prompting Strategies


Dataset: `dev`

**2.1.1.1 Zero-shot**


Direct instruction to extract entities without examples, relying solely on the model's pre-trained knowledge.


**Model:** `mistral:7b` (via Ollama)


**LLM Query:**

```txt
You are an expert in Named Entity Recognition (NER) for Vietnamese text.

TASK: Extract ALL named entities that appear DIRECTLY in the text. Categorize them into 3 types:

1. PERSON (People names):
   - Full names: "Nguyễn Văn A", "Đỗ Tất Lợi", "Angelina Jolie"
   - Titles + names: "ông A", "bà B", "anh Hùng", "chị Mai", "Mr. Kim"
   - Stage names/nicknames: "Hòa Minzy", "Đức Phúc"
   - Character names: "Bao Thanh Thiên", "Bao Chửng", "Kim Siêu Quần"
   - Alternate mentions: "Porzingis", "K. Porzingis", "Kristaps Porzingis" (all are same person)
   - Foreign names: "Brad Pitt", "Donald Trump", "Angela Merkel"
   - Author credits: Extract names after dashes or "Theo"

2. ORGANIZATIONS (Institutions, companies, teams):
   - Government agencies: "Bộ GD-ĐT", "Bộ Giáo dục và Đào tạo", "UBND tỉnh"
   - Companies: "CTCP Dầu thực vật Tường An", "TAC", "Oracle"
   - Schools: "THCS Kỳ Thượng", "Trường Tiểu học Kỳ Thịnh 2"
   - Sports teams/clubs: "Manchester United", "Hà Nội FC", "Quảng Nam", "Raptors"
   - International orgs: "WHO", "FIFA", "VFF", "UEFA", "EU"
   - Media: "phim trường Siêu Quần", "Báo Tiền phong", "VnExpress"
   - Note: Team names often appear without "FC" or "CLB" prefix

3. ADDRESS (Locations, places):
   - Countries: "Việt Nam", "Mỹ", "Trung Quốc", "Brazil"
   - Provinces/cities: "tỉnh Hà Tĩnh", "TP.HCM", "Hà Nội", "Thanh Đảo"
   - Districts: "quận 3", "huyện Kỳ Anh", "thị xã Kỳ Anh"
   - Wards/communes: "phường Kỳ Trinh", "xã Kỳ Nam"
   - Specific places: "sân Tam Kỳ", "Biển Đen", "đường Bàn Cờ", "Air Canada Centre"
   - Note: Vietnamese uses prefixes like "tỉnh", "thành phố", "quận", "huyện"

CRITICAL RULES:
✓ ONLY extract entities that appear DIRECTLY in the text (NO inference, NO translation)
✓ PRESERVE original spelling, accents, and capitalization from source text
✓ If same person has multiple mentions (e.g., "Đức Phúc" and "Phúc") → list ALL
✓ Each entity appears ONCE per category (remove duplicates)
✓ If no entities found → return empty array []
✓ Remove trailing punctuation: "ông A." → "ông A"
✓ Include ALL alternate names for same entity

VIETNAMESE TEXT TO ANALYZE:
Anh đi khắp núi, khắp ngàn/Không đâu đẹp bằng đá Bàn, sông Giăng, đó là câu ca về dòng sông nơi miền tây xứ Nghệ, chảy qua huyện Con Cuông mà ai cũng phải nhắc đến khi có dịp tới đây. Bên cạnh thiên nhiên nguyên sơ tuyệt đẹp, hai bên bờ sông với những cánh rừng nguyên sinh Vườn quốc gia Pù Mát, du lịch cộng đồng tại các bản làng dân tộc thiểu số ở Con Cuông đang tạo nên sự khởi sắc trong phát triển kinh tế gắn với bảo tồn văn hóa. Tháng 8 ở vùng cao. Dường như gió tây bớt gay gắt hơn, nhưng nắng vẫn còn chói chang. Sông Giăng nước trong xanh và có thể nhìn tận đáy. Chúng tôi dừng chân ở đập nước Phà Lài giữa núi rừng Môn Sơn, xã biên giới giáp nước bạn Lào, cách trung tâm thị trấn Con Cuông (huyện Con Cuông, Nghệ An) chưa đầy 20 km, để xuống bến thuyền, bắt đầu hành trình khám phá dòng sông Giăng, con sông dài nằm trong quần thể Vườn quốc gia Pù Mát. Ðược mở cửa đón khách du lịch từ hơn chục năm nay, nhưng giờ các dịch vụ du lịch ở khu vực này vẫn còn nghèo nàn. Thời điểm này, khách du lịch thưa vắng hơn độ tháng 5, tháng 6, vào mùa cao điểm du lịch. Người lái thuyền nhiều kinh nghiệm vượt thác sông Giăng đón chúng tôi lên những chiếc thuyền gỗ mang dáng độc mộc, gắn động cơ máy nổ công suất lớn. Ông nhắc mọi người mặc áo phao và kể với chúng tôi về dòng sông đã trở thành con đường nước đưa khách du lịch vào Vườn quốc gia Pù Mát. Ðó cũng là con đường mang lại không ít thu nhập trong cuộc mưu sinh của người dân địa phương. Chúng tôi ngồi thuyền mất khoảng hai tiếng để đi sâu vào vùng lõi của vườn quốc gia, nơi có các bản của tộc người Ðan Lai sinh sống. Chặng đường sông dài 20 km có lúc là trải nghiệm cảm giác mạnh khi ngược dòng nước với đầy sỏi to, sỏi nhỏ, đá cuội dưới lòng sông. Bù lại, những cung bậc cảm xúc thích thú, hồi hộp lại đan xen, thay đổi theo từng khúc địa hình của dòng sông. Và cả những phút lãng mạn ngắm rừng cây cổ thụ mọc trên vách đá xõa tán xuống hai bờ, những ngôi nhà sàn đơn sơ nằm cách xa nhau, giữa những ruộng ngô, ruộng lúa xanh tươi. Tiếp nối hành trình khám phá sông Giăng, đoàn lên bản Cò Phạt thuộc xã Môn Sơn nằm giữa vùng lõi Vườn quốc gia Pù Mát, vào thời gian sau vụ thu hoạch lúa của người Ðan Lai. Hai bên đường vào bản, nhìn những chân ruộng vừa gặt, thẳng đều tăm tắp, ít ai biết, chỉ hơn chục năm trước, cây lúa nước vẫn còn quá xa lạ với người Ðan Lai. Trước đây, người Ðan Lai sinh sống chủ yếu bằng săn bắt, họ chỉ biết vào rừng bẻ măng, bắt con thú, xuống sông tìm con cá, con tôm. Nhờ cán bộ dưới huyện lên hướng dẫn, chỉ bảo, bà con biết khai hoang đất làm ruộng, trồng cây lúa, cây ngô; chăn nuôi gà, lợn, trâu. Giờ nhà nào cũng đã có gạo ăn, hạn chế lên rừng kiếm cái ăn. So với Cò Phạt mười năm trước, giờ bản có nhiều đổi thay. Không còn cảnh trẻ em nhìn khách lạ sợ hãi, khép nép nữa. Bản mới được đầu tư xây dựng Ðiểm trường bản Cò Phạt thuộc Trường tiểu học 3 Môn Sơn với một lớp mầm non và năm lớp bậc tiểu học. Nhưng đời sống của đồng bào Ðan Lai còn gặp rất nhiều khó khăn, là lý do mà số học sinh theo lên học THCS ngày càng ít. Bản cũng nhận được sự quan tâm, hỗ trợ từ nhiều đoàn khách đến tham quan…, Phó Chủ tịch xã Môn Sơn Lương Văn Tuấn bày tỏ. Vườn quốc gia Pù Mát được thành lập năm 2002, với vùng bảo tồn rộng 94 nghìn ha và vùng đệm 100 nghìn ha, thuộc ba huyện Anh Sơn, Con Cuông, Tương Dương. Cách đây mười năm, Vườn được UNESCO công nhận là Khu dự trữ sinh quyển thế giới, nổi tiếng không chỉ ở sự phong phú, đa dạng của các loài thực, động vật hoang dã, quý hiếm, mà còn là địa chỉ du lịch sinh thái hấp dẫn du khách trong và ngoài nước. Vài năm trở lại đây, sự liên kết chặt chẽ giữa các địa phương, Ban quản lý Vườn quốc gia Pù Mát, các công ty lữ hành... tạo nên những tua trải nghiệm du lịch sinh thái, du lịch mạo hiểm và khám phá thú vị. Trưởng phòng Văn hóa huyện Con Cuông Nguyễn Xuân Nam chia sẻ: Thế mạnh để phát triển du lịch sinh thái là gắn với nhiều điểm đến kỳ thú nằm trong quần thể Vườn quốc gia như: thác Khe Kèm, khe nước Mọc, hang Thẳm Ồm, di tích lịch sử, văn hóa bia Ma Nhai, thành cổ Trà Lân.... Trải nghiệm cảnh sắc thiên nhiên nguyên sơ ở Khe Kèm. Nằm trong địa bàn Vườn quốc gia Pù Mát, những bản làng thuần nông của đồng bào dân tộc Thái ở Con Cuông vẫn giữ được những bản sắc văn hóa truyền thống độc đáo. Giữa mầu xanh của núi đồi, ruộng nương, đường liên thôn, liên xã được bê-tông hóa, bản làng phong quang sạch đẹp, thêm nhiều nếp nhà sàn mái ngói đỏ tươi, nhiều nhà sàn đã trở thành điểm lưu trú (homestay) cho khách du lịch. Theo tấm biển bằng tiếng Việt và tiếng Anh chỉ dẫn vào homestay Hoa Thụ, một trong ba hộ ở bản Nưa, xã Yên Khê tham gia đầu tiên mô hình du lịch cộng đồng, chúng tôi được chị Lô Thị Hoa, 43 tuổi, chủ nhân ra đầu ngõ đón. Từ năm 2011, chị Hoa quyết định bán cả bầy trâu, vay thêm ngân hàng, mạnh dạn sửa chữa, nâng cấp nhà sàn, bếp, với chi phí đến 200 triệu đồng. Tiền dành dụm được vợ chồng chị đầu tư cải tạo hệ thống vệ sinh khép kín sạch sẽ, mua sắm giường chiếu, chăn màn đầy đủ cho điểm homestay của mình. Ngôi nhà sàn gỗ mới sơn nằm xinh xắn trong khuôn viên bao quanh là những hàng cây xanh mọc đan chéo nhau, được chủ nhân cắt tỉa, chăm sóc, gìn giữ từ chục năm nay. Bản Nưa của xã Yên Khê có khoảng 146 hộ với hơn 600 dân, phần đông là người dân tộc Thái, đây là một trong những bản đi đầu trong phong trào xây dựng nông thôn mới. Ðể giữ chân du khách đến và ở lại với Bản Nưa, ngay từ khi bắt đầu làm du lịch cộng đồng, trong kế hoạch của các gia đình thì việc đầu tiên làm là chỉnh sửa, đầu tư xử lý và quy hoạch hệ thống nhà vệ sinh, chuồng chăn nuôi. Ðây là điều không dễ làm đối với hộ đồng bào Thái vốn quen với nếp sống cũ như nhà vệ sinh, chuồng chăn nuôi gắn liền hoặc gần nhà ở. Từ hai năm nay, ba homestay của Bản Nưa với sự hỗ trợ của các chuyên gia Nhật Bản trở thành những hộ điển hình làm du lịch cộng đồng, năng động, linh hoạt trong việc đón khách. Homestay của chị Lô Thị Hoa nằm trong số đó, đón 900 lượt khách trong nước và quốc tế vào năm ngoái và đón khoảng 1.200 lượt chỉ từ đầu năm tới nay. Ðây chính là thành công của dự án Ða dạng hóa sinh kế dựa vào du lịch di sản ở các làng nông, ngư nghiệp tỉnh Nghệ An với sự hỗ trợ của Cơ quan hợp tác quốc tế Nhật Bản (JICA). Nhằm giúp các hộ kinh doanh du lịch cộng đồng có thêm kiến thức phục vụ khách du lịch tại homestay, huyện Con Cuông cũng phối hợp với JICA tổ chức các lớp tập huấn về dịch vụ đón khách, về nấu ăn, tổ chức tham quan học tập kinh nghiệm tại Hòa Bình; giúp đỡ triển khai du lịch trải nghiệm tại các vườn cam... Từ khi mô hình du lịch cộng đồng tại Con Cuông thu hút khách du lịch, bà con dân tộc Thái nơi đây tham gia tích cực hơn vào các hoạt động sinh hoạt cộng đồng. Các nhóm, câu lạc bộ nấu ăn, múa hát được chính người dân thành lập để tôn vinh những bản sắc văn hóa của bản làng. Việc thành lập hợp tác xã thủ công mỹ nghệ xã Môn Sơn đóng tại bản Xiềng, sản xuất sản phẩm thổ cẩm được người nước ngoài ưa chuộng không chỉ góp phần gìn giữ nghề truyền thống, mà còn ngày càng tạo thêm việc làm, tăng thu nhập cho chị em trong bản. Bản Xiềng và điểm giới thiệu, bán sản phẩm của hợp tác xã nằm ngay trục đường chính của xã mới được tỉnh Nghệ An, huyện Con Cuông quan tâm đầu tư còn là điểm tham quan thú vị. Phó Giám đốc Sở Du lịch Nghệ An Nguyễn Mạnh Lợi cho rằng, du lịch huyện Con Cuông đã có nhiều khởi sắc nhờ sự đa dạng trong các tua du lịch, sự kết nối với các hợp tác xã, làng nghề truyền thống, tổ hợp nông nghiệp phục vụ du lịch cộng đồng. Mô hình vườn cam sinh thái tại bản Pha, xã Yên Khê với tổ hợp tác sản xuất các sản phẩm từ cam đang nâng giá trị của cây cam, mang lại hiệu quả kinh tế cho các hộ dân. Với sự hỗ trợ và hướng dẫn của JICA và Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam, các sản phẩm được chế biến từ quả cam sạch mở ra triển vọng mới về các sản phẩm du lịch địa phương. Anh An-đô Kát-su-hi-rô, chuyên gia dự án nêu trên, người trực tiếp hướng dẫn mô hình này chia sẻ, giờ đây bà con có thể tự sản xuất các sản phẩm từ quả cam vườn như rượu men cam, rượu hương cam, xà-phòng cam, mứt vỏ cam... cho khách du lịch. Năm 2016, khách đến tham quan Con Cuông tăng đột biến với hơn 35 nghìn lượt người so với lượng khách bình quân khoảng 10 nghìn lượt mỗi năm của giai đoạn 5 năm trước. Con Cuông cũng đón hơn bảy nghìn lượt khách riêng trong sáu tháng đầu năm 2017 và hơn tám nghìn lượt chỉ trong ba tháng gần đây. Những con số mà Phó Chủ tịch UBND huyện Con Cuông Kha Thị Tím chia sẻ ấy là để khẳng định rằng phát triển làng văn hóa du lịch cộng đồng đang là một hướng đi mới, đầy tiềm năng trong việc giúp người dân ở huyện miền núi xóa đói, giảm nghèo. Con Cuông được kỳ vọng trở thành một trong những trung tâm du lịch của miền tây Nghệ An. Hướng tới phát triển du lịch bền vững, hiện nay, kế hoạch của huyện là quy hoạch, đẩy mạnh xúc tiến xây dựng kết cấu hạ tầng phục vụ du lịch; xây dựng các tua, tuyến, điểm du lịch gắn kết với các tuyến du lịch của tỉnh. Đêm ở Bản Nưa. Giai điệu ngọt ngào của dân ca hay các điệu múa xòe, múa quạt, múa lăm vông… rộn rã trong tiếng cồng, tiếng chiêng dưới mái nhà sàn. Những cô gái Thái sáng đến lo chuyện làm vườn, chuyện đồng áng, tối về duyên dáng bên ánh lửa bập bùng. Tiếng nhạc quyện hòa tiếng hát mộc mạc ngân vang. Ðêm rượu cần bản làng mình, ai chưa say là chưa vui, ngôi nhà sàn nghiêng ngả, chuếnh choáng trong mặt người. Chum rượu nồng khao khát, niềm vui ta vơi đầy... Nào ta vít cần trúc, uống niềm vui vui cho say, niềm vui càng càng san sẻ, chớ để say say một người…, tiếng hát của các cô gái Thái cứ thế dặt dìu, quấn quýt, như tiếng lòng hò hẹn, như lời mời gọi đến với Con Cuông, gần gũi và chân tình. Bài và ảnh: TRÀ MY

RETURN ONLY THIS JSON (no explanations, no markdown):
{
  "person": [],
  "organizations": [],
  "address": []
}
```


**Result:**

```json
{
  "person": [
    "Lô Thị Hoa",
    "Phó Chủ tịch xã Môn Sơn Lương Văn Tuấn",
    "Trưởng phòng Văn hóa huyện Con Cuông Nguyễn Xuân Nam",
    "Phó Giám đốc Sở Du lịch Nghệ An Nguyễn Mạnh Lợi",
    "Phó Chủ tịch UBND huyện Con Cuông Kha Thị Tím",
    "Anh An-đô Kát-su-hi-rô"
  ],
  "organizations": [
    "Vườn quốc gia Pù Mát",
    "Trường tiểu học 3 Môn Sơn",
    "Hợp tác xã thủ công mỹ nghệ xã Môn Sơn",
    "Cơ quan hợp tác quốc tế Nhật Bản (JICA)",
    "Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam",
    "UBND huyện Con Cuông",
    "Sở Du lịch Nghệ An",
    "Ban quản lý Vườn quốc gia Pù Mát",
    "Trường Tiểu học Kỳ Thịnh 2",
    "Trường Tiểu học 3 Môn Sơn (mầm non và tiểu học)"
  ],
  "address": [
    "miền tây xứ Nghệ",
    "huyện Con Cuông",
    "Nghệ An",
    "đập nước Phà Lài",
    "xã Môn Sơn",
    "thị trấn Con Cuông",
    "Vườn quốc gia Pù Mát",
    "quần thể Vườn quốc gia Pù Mát",
    "bản Cò Phạt",
    "bản Nưa",
    "xã Yên Khê",
    "bản Xiềng",
    "bản Pha",
    "đường liên thôn, liên xã",
    "địa bàn Vườn quốc gia Pù Mát",
    "huyện Anh Sơn",
    "huyện Tương Dương",
    "đường Bê-tông hóa",
    "homestay Hoa Thụ",
    "bản làng",
    "bản Nưa",
    "bản Xiềng",
    "bản Pha",
    "xã Yên Khê",
    "xã Môn Sơn",
    "huyện Con Cuông",
    "tỉnh Nghệ An"
  ]
}
```


**2.1.1.2 Few-shot**

Provides 3 carefully selected examples demonstrating the task before processing new text.

**Model:** `mistral:7b` (via Ollama)

**LLM Query:**

```txt
You are an expert in Named Entity Recognition (NER) for Vietnamese text.

LEARN FROM THESE EXAMPLES showing correct Vietnamese entity extraction:


EXAMPLE 1:
Vietnamese Text: Theo Le Monde, tối 21-9, giờ địa phương, nữ tỷ phú giàu nhất thế giới Liliane Bettencourt đã qua đời trong một giấc ngủ yên bình tại nhà riêng ở Neuilly-sur-Seine (Pháp), thọ 94 tuổi. Liliane Bettencourt là con duy nhất của nhà hóa học Eugene Schueller, người đã sáng lập tập đoàn mỹ phẩm danh tiếng của Pháp L'Oreal từ hồi đầu thế kỷ 20. Bà Liliane Bettencourt thừa kế tập đoàn L'Oreal năm 1957 sau khi người cha qua đời, trở thành người phụ nữ giàu nhất thế giới và nằm trong top 20 người giàu có n
Extracted Entities:
{
  "person": [
    "Liliane Bettencourt",
    "Eugene Schueller"
  ],
  "organizations": [
    "Le Monde",
    "L'Oreal",
    "Forbes"
  ],
  "address": [
    "Neuilly-sur-Seine",
    "Pháp"
  ]
}
---

EXAMPLE 2:
Vietnamese Text: Eugenia Silva sinh ngày 13/01/1976 tại Madrid, Tây Ban Nha. Cô sở hữu chiều cao 1m77, và là nhân vật có tiếng tăm tại thành phố thủ đô. Sinh ra trong một gia đình có truyền thống ở Madrid, Eugenia Silva dù là người mẫu nhưng cũng đã tốt nghiệp luật sư. Năm mới 16 tuổi, cô đã chiến thắng trong thi siêu mẫu chung kết năm do tạp chí Elite tổ chức. Đây cũng là bước khởi đầu cho sự nghiệp của người phụ nữ này. Châu Tinh - Thể thao Việt Nam | 20:35 22/09/2017
Extracted Entities:
{
  "person": [
    "Eugenia Silva",
    "Châu Tinh"
  ],
  "organizations": [
    "Elite"
  ],
  "address": [
    "Madrid",
    "Tây Ban Nha",
    "Việt Nam"
  ]
}
---

EXAMPLE 3:
Vietnamese Text: Đoạn video ghi lại cảnh quân đội Nga tấn công ác liệt nhóm liên minh thánh chiến Hayat Tahrir al-Sham tại tỉnh Idlib, Syria. South Front dẫn nguồn tin từ Bộ Quốc phòng Nga cho biết, quân đội Nga đã tấn công ác liệt nhóm liên minh thánh chiến Hayat Tahrir al-Sham tại khu vực giảm xung đột ở tỉnh Idlib. Một căn cứ của liên minh thánh chiến Hayat Tahrir al-Sham nổ tung ở Idlib. Ảnh: South Front. Nguồn tin cho hay, lực lượng đặc nhiệm Nga thực hiện nhiệm vụ giải vây cho một trung đội quân cảnh Nga đ
Extracted Entities:
{
  "person": [
    "Thiên An"
  ],
  "organizations": [
    "Hayat Tahrir al-Sham"
  ],
  "address": [
    "Nga",
    "tỉnh Idlib",
    "Syria",
    "Idlib"
  ]
}
---


KEY PATTERNS TO RECOGNIZE:
1. PERSON: Vietnamese names (3+ words), foreign names, titles (ông/bà/anh/chị), stage names, character names
2. ORGANIZATIONS:
   - Vietnamese gov agencies start with "Bộ", "UBND", "CTCP"
   - Schools often have "Trường", "THCS", "THPT"
   - Sports teams may appear as just city names: "Quảng Nam", "Hà Nội"
3. ADDRESS:
   - Administrative hierarchy: "tỉnh" (province), "thành phố" (city), "quận" (district), "huyện" (district), "xã" (commune), "phường" (ward)
   - Always keep the prefix with the location name

EXTRACTION RULES:
✓ Extract ONLY entities that appear DIRECTLY in the text
✓ PRESERVE original Vietnamese spelling with accents (dấu)
✓ List ALL alternate mentions of same entity separately
✓ Remove duplicates within each category
✓ Keep original capitalization from text

NOW EXTRACT ENTITIES FROM THIS NEW TEXT:
Anh đi khắp núi, khắp ngàn/Không đâu đẹp bằng đá Bàn, sông Giăng, đó là câu ca về dòng sông nơi miền tây xứ Nghệ, chảy qua huyện Con Cuông mà ai cũng phải nhắc đến khi có dịp tới đây. Bên cạnh thiên nhiên nguyên sơ tuyệt đẹp, hai bên bờ sông với những cánh rừng nguyên sinh Vườn quốc gia Pù Mát, du lịch cộng đồng tại các bản làng dân tộc thiểu số ở Con Cuông đang tạo nên sự khởi sắc trong phát triển kinh tế gắn với bảo tồn văn hóa. Tháng 8 ở vùng cao. Dường như gió tây bớt gay gắt hơn, nhưng nắng vẫn còn chói chang. Sông Giăng nước trong xanh và có thể nhìn tận đáy. Chúng tôi dừng chân ở đập nước Phà Lài giữa núi rừng Môn Sơn, xã biên giới giáp nước bạn Lào, cách trung tâm thị trấn Con Cuông (huyện Con Cuông, Nghệ An) chưa đầy 20 km, để xuống bến thuyền, bắt đầu hành trình khám phá dòng sông Giăng, con sông dài nằm trong quần thể Vườn quốc gia Pù Mát. Ðược mở cửa đón khách du lịch từ hơn chục năm nay, nhưng giờ các dịch vụ du lịch ở khu vực này vẫn còn nghèo nàn. Thời điểm này, khách du lịch thưa vắng hơn độ tháng 5, tháng 6, vào mùa cao điểm du lịch. Người lái thuyền nhiều kinh nghiệm vượt thác sông Giăng đón chúng tôi lên những chiếc thuyền gỗ mang dáng độc mộc, gắn động cơ máy nổ công suất lớn. Ông nhắc mọi người mặc áo phao và kể với chúng tôi về dòng sông đã trở thành con đường nước đưa khách du lịch vào Vườn quốc gia Pù Mát. Ðó cũng là con đường mang lại không ít thu nhập trong cuộc mưu sinh của người dân địa phương. Chúng tôi ngồi thuyền mất khoảng hai tiếng để đi sâu vào vùng lõi của vườn quốc gia, nơi có các bản của tộc người Ðan Lai sinh sống. Chặng đường sông dài 20 km có lúc là trải nghiệm cảm giác mạnh khi ngược dòng nước với đầy sỏi to, sỏi nhỏ, đá cuội dưới lòng sông. Bù lại, những cung bậc cảm xúc thích thú, hồi hộp lại đan xen, thay đổi theo từng khúc địa hình của dòng sông. Và cả những phút lãng mạn ngắm rừng cây cổ thụ mọc trên vách đá xõa tán xuống hai bờ, những ngôi nhà sàn đơn sơ nằm cách xa nhau, giữa những ruộng ngô, ruộng lúa xanh tươi. Tiếp nối hành trình khám phá sông Giăng, đoàn lên bản Cò Phạt thuộc xã Môn Sơn nằm giữa vùng lõi Vườn quốc gia Pù Mát, vào thời gian sau vụ thu hoạch lúa của người Ðan Lai. Hai bên đường vào bản, nhìn những chân ruộng vừa gặt, thẳng đều tăm tắp, ít ai biết, chỉ hơn chục năm trước, cây lúa nước vẫn còn quá xa lạ với người Ðan Lai. Trước đây, người Ðan Lai sinh sống chủ yếu bằng săn bắt, họ chỉ biết vào rừng bẻ măng, bắt con thú, xuống sông tìm con cá, con tôm. Nhờ cán bộ dưới huyện lên hướng dẫn, chỉ bảo, bà con biết khai hoang đất làm ruộng, trồng cây lúa, cây ngô; chăn nuôi gà, lợn, trâu. Giờ nhà nào cũng đã có gạo ăn, hạn chế lên rừng kiếm cái ăn. So với Cò Phạt mười năm trước, giờ bản có nhiều đổi thay. Không còn cảnh trẻ em nhìn khách lạ sợ hãi, khép nép nữa. Bản mới được đầu tư xây dựng Ðiểm trường bản Cò Phạt thuộc Trường tiểu học 3 Môn Sơn với một lớp mầm non và năm lớp bậc tiểu học. Nhưng đời sống của đồng bào Ðan Lai còn gặp rất nhiều khó khăn, là lý do mà số học sinh theo lên học THCS ngày càng ít. Bản cũng nhận được sự quan tâm, hỗ trợ từ nhiều đoàn khách đến tham quan…, Phó Chủ tịch xã Môn Sơn Lương Văn Tuấn bày tỏ. Vườn quốc gia Pù Mát được thành lập năm 2002, với vùng bảo tồn rộng 94 nghìn ha và vùng đệm 100 nghìn ha, thuộc ba huyện Anh Sơn, Con Cuông, Tương Dương. Cách đây mười năm, Vườn được UNESCO công nhận là Khu dự trữ sinh quyển thế giới, nổi tiếng không chỉ ở sự phong phú, đa dạng của các loài thực, động vật hoang dã, quý hiếm, mà còn là địa chỉ du lịch sinh thái hấp dẫn du khách trong và ngoài nước. Vài năm trở lại đây, sự liên kết chặt chẽ giữa các địa phương, Ban quản lý Vườn quốc gia Pù Mát, các công ty lữ hành... tạo nên những tua trải nghiệm du lịch sinh thái, du lịch mạo hiểm và khám phá thú vị. Trưởng phòng Văn hóa huyện Con Cuông Nguyễn Xuân Nam chia sẻ: Thế mạnh để phát triển du lịch sinh thái là gắn với nhiều điểm đến kỳ thú nằm trong quần thể Vườn quốc gia như: thác Khe Kèm, khe nước Mọc, hang Thẳm Ồm, di tích lịch sử, văn hóa bia Ma Nhai, thành cổ Trà Lân.... Trải nghiệm cảnh sắc thiên nhiên nguyên sơ ở Khe Kèm. Nằm trong địa bàn Vườn quốc gia Pù Mát, những bản làng thuần nông của đồng bào dân tộc Thái ở Con Cuông vẫn giữ được những bản sắc văn hóa truyền thống độc đáo. Giữa mầu xanh của núi đồi, ruộng nương, đường liên thôn, liên xã được bê-tông hóa, bản làng phong quang sạch đẹp, thêm nhiều nếp nhà sàn mái ngói đỏ tươi, nhiều nhà sàn đã trở thành điểm lưu trú (homestay) cho khách du lịch. Theo tấm biển bằng tiếng Việt và tiếng Anh chỉ dẫn vào homestay Hoa Thụ, một trong ba hộ ở bản Nưa, xã Yên Khê tham gia đầu tiên mô hình du lịch cộng đồng, chúng tôi được chị Lô Thị Hoa, 43 tuổi, chủ nhân ra đầu ngõ đón. Từ năm 2011, chị Hoa quyết định bán cả bầy trâu, vay thêm ngân hàng, mạnh dạn sửa chữa, nâng cấp nhà sàn, bếp, với chi phí đến 200 triệu đồng. Tiền dành dụm được vợ chồng chị đầu tư cải tạo hệ thống vệ sinh khép kín sạch sẽ, mua sắm giường chiếu, chăn màn đầy đủ cho điểm homestay của mình. Ngôi nhà sàn gỗ mới sơn nằm xinh xắn trong khuôn viên bao quanh là những hàng cây xanh mọc đan chéo nhau, được chủ nhân cắt tỉa, chăm sóc, gìn giữ từ chục năm nay. Bản Nưa của xã Yên Khê có khoảng 146 hộ với hơn 600 dân, phần đông là người dân tộc Thái, đây là một trong những bản đi đầu trong phong trào xây dựng nông thôn mới. Ðể giữ chân du khách đến và ở lại với Bản Nưa, ngay từ khi bắt đầu làm du lịch cộng đồng, trong kế hoạch của các gia đình thì việc đầu tiên làm là chỉnh sửa, đầu tư xử lý và quy hoạch hệ thống nhà vệ sinh, chuồng chăn nuôi. Ðây là điều không dễ làm đối với hộ đồng bào Thái vốn quen với nếp sống cũ như nhà vệ sinh, chuồng chăn nuôi gắn liền hoặc gần nhà ở. Từ hai năm nay, ba homestay của Bản Nưa với sự hỗ trợ của các chuyên gia Nhật Bản trở thành những hộ điển hình làm du lịch cộng đồng, năng động, linh hoạt trong việc đón khách. Homestay của chị Lô Thị Hoa nằm trong số đó, đón 900 lượt khách trong nước và quốc tế vào năm ngoái và đón khoảng 1.200 lượt chỉ từ đầu năm tới nay. Ðây chính là thành công của dự án Ða dạng hóa sinh kế dựa vào du lịch di sản ở các làng nông, ngư nghiệp tỉnh Nghệ An với sự hỗ trợ của Cơ quan hợp tác quốc tế Nhật Bản (JICA). Nhằm giúp các hộ kinh doanh du lịch cộng đồng có thêm kiến thức phục vụ khách du lịch tại homestay, huyện Con Cuông cũng phối hợp với JICA tổ chức các lớp tập huấn về dịch vụ đón khách, về nấu ăn, tổ chức tham quan học tập kinh nghiệm tại Hòa Bình; giúp đỡ triển khai du lịch trải nghiệm tại các vườn cam... Từ khi mô hình du lịch cộng đồng tại Con Cuông thu hút khách du lịch, bà con dân tộc Thái nơi đây tham gia tích cực hơn vào các hoạt động sinh hoạt cộng đồng. Các nhóm, câu lạc bộ nấu ăn, múa hát được chính người dân thành lập để tôn vinh những bản sắc văn hóa của bản làng. Việc thành lập hợp tác xã thủ công mỹ nghệ xã Môn Sơn đóng tại bản Xiềng, sản xuất sản phẩm thổ cẩm được người nước ngoài ưa chuộng không chỉ góp phần gìn giữ nghề truyền thống, mà còn ngày càng tạo thêm việc làm, tăng thu nhập cho chị em trong bản. Bản Xiềng và điểm giới thiệu, bán sản phẩm của hợp tác xã nằm ngay trục đường chính của xã mới được tỉnh Nghệ An, huyện Con Cuông quan tâm đầu tư còn là điểm tham quan thú vị. Phó Giám đốc Sở Du lịch Nghệ An Nguyễn Mạnh Lợi cho rằng, du lịch huyện Con Cuông đã có nhiều khởi sắc nhờ sự đa dạng trong các tua du lịch, sự kết nối với các hợp tác xã, làng nghề truyền thống, tổ hợp nông nghiệp phục vụ du lịch cộng đồng. Mô hình vườn cam sinh thái tại bản Pha, xã Yên Khê với tổ hợp tác sản xuất các sản phẩm từ cam đang nâng giá trị của cây cam, mang lại hiệu quả kinh tế cho các hộ dân. Với sự hỗ trợ và hướng dẫn của JICA và Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam, các sản phẩm được chế biến từ quả cam sạch mở ra triển vọng mới về các sản phẩm du lịch địa phương. Anh An-đô Kát-su-hi-rô, chuyên gia dự án nêu trên, người trực tiếp hướng dẫn mô hình này chia sẻ, giờ đây bà con có thể tự sản xuất các sản phẩm từ quả cam vườn như rượu men cam, rượu hương cam, xà-phòng cam, mứt vỏ cam... cho khách du lịch. Năm 2016, khách đến tham quan Con Cuông tăng đột biến với hơn 35 nghìn lượt người so với lượng khách bình quân khoảng 10 nghìn lượt mỗi năm của giai đoạn 5 năm trước. Con Cuông cũng đón hơn bảy nghìn lượt khách riêng trong sáu tháng đầu năm 2017 và hơn tám nghìn lượt chỉ trong ba tháng gần đây. Những con số mà Phó Chủ tịch UBND huyện Con Cuông Kha Thị Tím chia sẻ ấy là để khẳng định rằng phát triển làng văn hóa du lịch cộng đồng đang là một hướng đi mới, đầy tiềm năng trong việc giúp người dân ở huyện miền núi xóa đói, giảm nghèo. Con Cuông được kỳ vọng trở thành một trong những trung tâm du lịch của miền tây Nghệ An. Hướng tới phát triển du lịch bền vững, hiện nay, kế hoạch của huyện là quy hoạch, đẩy mạnh xúc tiến xây dựng kết cấu hạ tầng phục vụ du lịch; xây dựng các tua, tuyến, điểm du lịch gắn kết với các tuyến du lịch của tỉnh. Đêm ở Bản Nưa. Giai điệu ngọt ngào của dân ca hay các điệu múa xòe, múa quạt, múa lăm vông… rộn rã trong tiếng cồng, tiếng chiêng dưới mái nhà sàn. Những cô gái Thái sáng đến lo chuyện làm vườn, chuyện đồng áng, tối về duyên dáng bên ánh lửa bập bùng. Tiếng nhạc quyện hòa tiếng hát mộc mạc ngân vang. Ðêm rượu cần bản làng mình, ai chưa say là chưa vui, ngôi nhà sàn nghiêng ngả, chuếnh choáng trong mặt người. Chum rượu nồng khao khát, niềm vui ta vơi đầy... Nào ta vít cần trúc, uống niềm vui vui cho say, niềm vui càng càng san sẻ, chớ để say say một người…, tiếng hát của các cô gái Thái cứ thế dặt dìu, quấn quýt, như tiếng lòng hò hẹn, như lời mời gọi đến với Con Cuông, gần gũi và chân tình. Bài và ảnh: TRÀ MY

RETURN ONLY JSON (no explanations, no markdown):
{"person": [], "organizations": [], "address": []}
```

**Result:**

```json
{
  "person": [
    "Lô Thị Hoa",
    "Nguyễn Xuân Nam",
    "Lương Văn Tuấn",
    "Nguyễn Mạnh Lợi",
    "Kha Thị Tím",
    "Anh An-đô Kát-su-hi-rô",
    "TRÀ MY"
  ],
  "organizations": [
    "Vườn quốc gia Pù Mát",
    "Trường tiểu học 3 Mòn Sơn",
    "Trường tiểu học 3 Môn Sơn",
    "Hợp tác xã thủ công mỹ nghệ xã Môn Sơn",
    "Cơ quan hợp tác quốc tế Nhật Bản (JICA)",
    "Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam",
    "Sở Du lịch Nghệ An",
    "UBND huyện Con Cuông",
    "Ban quản lý Vườn quốc gia Pù Mát"
  ],
  "address": [
    "Con Cuông",
    "Nghệ An",
    "Bản Nưa",
    "Xã Yên Khê",
    "Xã Môn Sơn",
    "Xã Yên Khê (Bản Pha)",
    "Bản Xiềng",
    "Khe Kèm",
    "Khe nước Mọc",
    "Hang Thẳm Ồm",
    "Di tích lịch sử, văn hóa bia Ma Nhai",
    "Thành cổ Trà Lân",
    "Bản Nưa (Xã Yên Khê)",
    "Bản Nưa (Homestay Hoa Thụ)",
    "Bản Xiềng (Hợp tác xã thủ công mỹ nghệ)",
    "Bản Pha (Vườn cam sinh thái)"
  ]
}
```

**2.1.1.3 Chain of Thought (CoT)**

Instructs the model to reason step-by-step before extracting entities.

**Model:** `mistral:7b` (via Ollama)

**LLM Query:**

```txt
You are an expert in Named Entity Recognition (NER) for Vietnamese text.
Analyze this Vietnamese text step by step to extract all named entities.

VIETNAMESE TEXT:
Anh đi khắp núi, khắp ngàn/Không đâu đẹp bằng đá Bàn, sông Giăng, đó là câu ca về dòng sông nơi miền tây xứ Nghệ, chảy qua huyện Con Cuông mà ai cũng phải nhắc đến khi có dịp tới đây. Bên cạnh thiên nhiên nguyên sơ tuyệt đẹp, hai bên bờ sông với những cánh rừng nguyên sinh Vườn quốc gia Pù Mát, du lịch cộng đồng tại các bản làng dân tộc thiểu số ở Con Cuông đang tạo nên sự khởi sắc trong phát triển kinh tế gắn với bảo tồn văn hóa. Tháng 8 ở vùng cao. Dường như gió tây bớt gay gắt hơn, nhưng nắng vẫn còn chói chang. Sông Giăng nước trong xanh và có thể nhìn tận đáy. Chúng tôi dừng chân ở đập nước Phà Lài giữa núi rừng Môn Sơn, xã biên giới giáp nước bạn Lào, cách trung tâm thị trấn Con Cuông (huyện Con Cuông, Nghệ An) chưa đầy 20 km, để xuống bến thuyền, bắt đầu hành trình khám phá dòng sông Giăng, con sông dài nằm trong quần thể Vườn quốc gia Pù Mát. Ðược mở cửa đón khách du lịch từ hơn chục năm nay, nhưng giờ các dịch vụ du lịch ở khu vực này vẫn còn nghèo nàn. Thời điểm này, khách du lịch thưa vắng hơn độ tháng 5, tháng 6, vào mùa cao điểm du lịch. Người lái thuyền nhiều kinh nghiệm vượt thác sông Giăng đón chúng tôi lên những chiếc thuyền gỗ mang dáng độc mộc, gắn động cơ máy nổ công suất lớn. Ông nhắc mọi người mặc áo phao và kể với chúng tôi về dòng sông đã trở thành con đường nước đưa khách du lịch vào Vườn quốc gia Pù Mát. Ðó cũng là con đường mang lại không ít thu nhập trong cuộc mưu sinh của người dân địa phương. Chúng tôi ngồi thuyền mất khoảng hai tiếng để đi sâu vào vùng lõi của vườn quốc gia, nơi có các bản của tộc người Ðan Lai sinh sống. Chặng đường sông dài 20 km có lúc là trải nghiệm cảm giác mạnh khi ngược dòng nước với đầy sỏi to, sỏi nhỏ, đá cuội dưới lòng sông. Bù lại, những cung bậc cảm xúc thích thú, hồi hộp lại đan xen, thay đổi theo từng khúc địa hình của dòng sông. Và cả những phút lãng mạn ngắm rừng cây cổ thụ mọc trên vách đá xõa tán xuống hai bờ, những ngôi nhà sàn đơn sơ nằm cách xa nhau, giữa những ruộng ngô, ruộng lúa xanh tươi. Tiếp nối hành trình khám phá sông Giăng, đoàn lên bản Cò Phạt thuộc xã Môn Sơn nằm giữa vùng lõi Vườn quốc gia Pù Mát, vào thời gian sau vụ thu hoạch lúa của người Ðan Lai. Hai bên đường vào bản, nhìn những chân ruộng vừa gặt, thẳng đều tăm tắp, ít ai biết, chỉ hơn chục năm trước, cây lúa nước vẫn còn quá xa lạ với người Ðan Lai. Trước đây, người Ðan Lai sinh sống chủ yếu bằng săn bắt, họ chỉ biết vào rừng bẻ măng, bắt con thú, xuống sông tìm con cá, con tôm. Nhờ cán bộ dưới huyện lên hướng dẫn, chỉ bảo, bà con biết khai hoang đất làm ruộng, trồng cây lúa, cây ngô; chăn nuôi gà, lợn, trâu. Giờ nhà nào cũng đã có gạo ăn, hạn chế lên rừng kiếm cái ăn. So với Cò Phạt mười năm trước, giờ bản có nhiều đổi thay. Không còn cảnh trẻ em nhìn khách lạ sợ hãi, khép nép nữa. Bản mới được đầu tư xây dựng Ðiểm trường bản Cò Phạt thuộc Trường tiểu học 3 Môn Sơn với một lớp mầm non và năm lớp bậc tiểu học. Nhưng đời sống của đồng bào Ðan Lai còn gặp rất nhiều khó khăn, là lý do mà số học sinh theo lên học THCS ngày càng ít. Bản cũng nhận được sự quan tâm, hỗ trợ từ nhiều đoàn khách đến tham quan…, Phó Chủ tịch xã Môn Sơn Lương Văn Tuấn bày tỏ. Vườn quốc gia Pù Mát được thành lập năm 2002, với vùng bảo tồn rộng 94 nghìn ha và vùng đệm 100 nghìn ha, thuộc ba huyện Anh Sơn, Con Cuông, Tương Dương. Cách đây mười năm, Vườn được UNESCO công nhận là Khu dự trữ sinh quyển thế giới, nổi tiếng không chỉ ở sự phong phú, đa dạng của các loài thực, động vật hoang dã, quý hiếm, mà còn là địa chỉ du lịch sinh thái hấp dẫn du khách trong và ngoài nước. Vài năm trở lại đây, sự liên kết chặt chẽ giữa các địa phương, Ban quản lý Vườn quốc gia Pù Mát, các công ty lữ hành... tạo nên những tua trải nghiệm du lịch sinh thái, du lịch mạo hiểm và khám phá thú vị. Trưởng phòng Văn hóa huyện Con Cuông Nguyễn Xuân Nam chia sẻ: Thế mạnh để phát triển du lịch sinh thái là gắn với nhiều điểm đến kỳ thú nằm trong quần thể Vườn quốc gia như: thác Khe Kèm, khe nước Mọc, hang Thẳm Ồm, di tích lịch sử, văn hóa bia Ma Nhai, thành cổ Trà Lân.... Trải nghiệm cảnh sắc thiên nhiên nguyên sơ ở Khe Kèm. Nằm trong địa bàn Vườn quốc gia Pù Mát, những bản làng thuần nông của đồng bào dân tộc Thái ở Con Cuông vẫn giữ được những bản sắc văn hóa truyền thống độc đáo. Giữa mầu xanh của núi đồi, ruộng nương, đường liên thôn, liên xã được bê-tông hóa, bản làng phong quang sạch đẹp, thêm nhiều nếp nhà sàn mái ngói đỏ tươi, nhiều nhà sàn đã trở thành điểm lưu trú (homestay) cho khách du lịch. Theo tấm biển bằng tiếng Việt và tiếng Anh chỉ dẫn vào homestay Hoa Thụ, một trong ba hộ ở bản Nưa, xã Yên Khê tham gia đầu tiên mô hình du lịch cộng đồng, chúng tôi được chị Lô Thị Hoa, 43 tuổi, chủ nhân ra đầu ngõ đón. Từ năm 2011, chị Hoa quyết định bán cả bầy trâu, vay thêm ngân hàng, mạnh dạn sửa chữa, nâng cấp nhà sàn, bếp, với chi phí đến 200 triệu đồng. Tiền dành dụm được vợ chồng chị đầu tư cải tạo hệ thống vệ sinh khép kín sạch sẽ, mua sắm giường chiếu, chăn màn đầy đủ cho điểm homestay của mình. Ngôi nhà sàn gỗ mới sơn nằm xinh xắn trong khuôn viên bao quanh là những hàng cây xanh mọc đan chéo nhau, được chủ nhân cắt tỉa, chăm sóc, gìn giữ từ chục năm nay. Bản Nưa của xã Yên Khê có khoảng 146 hộ với hơn 600 dân, phần đông là người dân tộc Thái, đây là một trong những bản đi đầu trong phong trào xây dựng nông thôn mới. Ðể giữ chân du khách đến và ở lại với Bản Nưa, ngay từ khi bắt đầu làm du lịch cộng đồng, trong kế hoạch của các gia đình thì việc đầu tiên làm là chỉnh sửa, đầu tư xử lý và quy hoạch hệ thống nhà vệ sinh, chuồng chăn nuôi. Ðây là điều không dễ làm đối với hộ đồng bào Thái vốn quen với nếp sống cũ như nhà vệ sinh, chuồng chăn nuôi gắn liền hoặc gần nhà ở. Từ hai năm nay, ba homestay của Bản Nưa với sự hỗ trợ của các chuyên gia Nhật Bản trở thành những hộ điển hình làm du lịch cộng đồng, năng động, linh hoạt trong việc đón khách. Homestay của chị Lô Thị Hoa nằm trong số đó, đón 900 lượt khách trong nước và quốc tế vào năm ngoái và đón khoảng 1.200 lượt chỉ từ đầu năm tới nay. Ðây chính là thành công của dự án Ða dạng hóa sinh kế dựa vào du lịch di sản ở các làng nông, ngư nghiệp tỉnh Nghệ An với sự hỗ trợ của Cơ quan hợp tác quốc tế Nhật Bản (JICA). Nhằm giúp các hộ kinh doanh du lịch cộng đồng có thêm kiến thức phục vụ khách du lịch tại homestay, huyện Con Cuông cũng phối hợp với JICA tổ chức các lớp tập huấn về dịch vụ đón khách, về nấu ăn, tổ chức tham quan học tập kinh nghiệm tại Hòa Bình; giúp đỡ triển khai du lịch trải nghiệm tại các vườn cam... Từ khi mô hình du lịch cộng đồng tại Con Cuông thu hút khách du lịch, bà con dân tộc Thái nơi đây tham gia tích cực hơn vào các hoạt động sinh hoạt cộng đồng. Các nhóm, câu lạc bộ nấu ăn, múa hát được chính người dân thành lập để tôn vinh những bản sắc văn hóa của bản làng. Việc thành lập hợp tác xã thủ công mỹ nghệ xã Môn Sơn đóng tại bản Xiềng, sản xuất sản phẩm thổ cẩm được người nước ngoài ưa chuộng không chỉ góp phần gìn giữ nghề truyền thống, mà còn ngày càng tạo thêm việc làm, tăng thu nhập cho chị em trong bản. Bản Xiềng và điểm giới thiệu, bán sản phẩm của hợp tác xã nằm ngay trục đường chính của xã mới được tỉnh Nghệ An, huyện Con Cuông quan tâm đầu tư còn là điểm tham quan thú vị. Phó Giám đốc Sở Du lịch Nghệ An Nguyễn Mạnh Lợi cho rằng, du lịch huyện Con Cuông đã có nhiều khởi sắc nhờ sự đa dạng trong các tua du lịch, sự kết nối với các hợp tác xã, làng nghề truyền thống, tổ hợp nông nghiệp phục vụ du lịch cộng đồng. Mô hình vườn cam sinh thái tại bản Pha, xã Yên Khê với tổ hợp tác sản xuất các sản phẩm từ cam đang nâng giá trị của cây cam, mang lại hiệu quả kinh tế cho các hộ dân. Với sự hỗ trợ và hướng dẫn của JICA và Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam, các sản phẩm được chế biến từ quả cam sạch mở ra triển vọng mới về các sản phẩm du lịch địa phương. Anh An-đô Kát-su-hi-rô, chuyên gia dự án nêu trên, người trực tiếp hướng dẫn mô hình này chia sẻ, giờ đây bà con có thể tự sản xuất các sản phẩm từ quả cam vườn như rượu men cam, rượu hương cam, xà-phòng cam, mứt vỏ cam... cho khách du lịch. Năm 2016, khách đến tham quan Con Cuông tăng đột biến với hơn 35 nghìn lượt người so với lượng khách bình quân khoảng 10 nghìn lượt mỗi năm của giai đoạn 5 năm trước. Con Cuông cũng đón hơn bảy nghìn lượt khách riêng trong sáu tháng đầu năm 2017 và hơn tám nghìn lượt chỉ trong ba tháng gần đây. Những con số mà Phó Chủ tịch UBND huyện Con Cuông Kha Thị Tím chia sẻ ấy là để khẳng định rằng phát triển làng văn hóa du lịch cộng đồng đang là một hướng đi mới, đầy tiềm năng trong việc giúp người dân ở huyện miền núi xóa đói, giảm nghèo. Con Cuông được kỳ vọng trở thành một trong những trung tâm du lịch của miền tây Nghệ An. Hướng tới phát triển du lịch bền vững, hiện nay, kế hoạch của huyện là quy hoạch, đẩy mạnh xúc tiến xây dựng kết cấu hạ tầng phục vụ du lịch; xây dựng các tua, tuyến, điểm du lịch gắn kết với các tuyến du lịch của tỉnh. Đêm ở Bản Nưa. Giai điệu ngọt ngào của dân ca hay các điệu múa xòe, múa quạt, múa lăm vông… rộn rã trong tiếng cồng, tiếng chiêng dưới mái nhà sàn. Những cô gái Thái sáng đến lo chuyện làm vườn, chuyện đồng áng, tối về duyên dáng bên ánh lửa bập bùng. Tiếng nhạc quyện hòa tiếng hát mộc mạc ngân vang. Ðêm rượu cần bản làng mình, ai chưa say là chưa vui, ngôi nhà sàn nghiêng ngả, chuếnh choáng trong mặt người. Chum rượu nồng khao khát, niềm vui ta vơi đầy... Nào ta vít cần trúc, uống niềm vui vui cho say, niềm vui càng càng san sẻ, chớ để say say một người…, tiếng hát của các cô gái Thái cứ thế dặt dìu, quấn quýt, như tiếng lòng hò hẹn, như lời mời gọi đến với Con Cuông, gần gũi và chân tình. Bài và ảnh: TRÀ MY

STEP 1: Understand the context
- What is this text about? (news, sports, entertainment, politics, etc.)
- This helps identify entity types (athletes vs actors vs politicians)

STEP 2: Find ALL PERSON entities
Scan for:
- Vietnamese names (usually 2-4 words, capitalized): "Nguyễn Văn A", "Đỗ Tất Lợi"
- Titles + names: "ông Kim", "bà Nghĩa", "anh Hùng", "chị Mai", "Mr. Smith"
- Names in quotes (often stage names or character names): "Bao Thanh Thiên", "Hòa Minzy"
- Foreign names in Latin script: "Brad Pitt", "LeBron James", "Angela Merkel"
- Abbreviated names: "Phúc" if "Đức Phúc" appears, "Kim" if "Kim Siêu Quần" appears
- Author credits (after dash or "Theo"): "- Thanh Hoài", "Theo TH"
IMPORTANT: If same person has multiple mentions → ADD ALL (e.g., both "Porzingis" AND "Kristaps Porzingis")

STEP 3: Find ALL ORGANIZATIONS
Look for Vietnamese patterns:
- Government: starts with "Bộ" (ministry), "UBND", "Liên đoàn", "Ủy ban"
- Companies: contains "CTCP", "Công ty", "Corporation"
- Schools: starts with "Trường", contains "THCS", "THPT", "Tiểu học"
- Sports teams: often just city names ("Hà Nội", "Quảng Nam") or with "FC", "CLB"
- Acronyms: WHO, FIFA, VFF, GD-ĐT, TAC, EU, NASA
- Media: "Báo" + name, magazine names

STEP 4: Find ALL ADDRESS entities
Vietnamese administrative hierarchy (keep prefixes):
- Province: "tỉnh" + name (e.g., "tỉnh Hà Tĩnh", "tỉnh Quảng Bình")
- City: "thành phố" or "TP" + name (e.g., "TP.HCM", "thành phố Hà Nội")
- District: "quận" or "huyện" or "thị xã" + name
- Ward/commune: "phường" or "xã" + name
- Countries: "Việt Nam", "Mỹ", "Trung Quốc", "Brazil"
- Specific places: stadiums ("sân" + name), seas ("Biển" + name), streets ("đường" + name)

STEP 5: Clean and deduplicate
- Remove exact duplicates from each category
- Keep original spelling/capitalization from source text
- Remove trailing punctuation (. , ; :)
- Do NOT guess or infer entities not in text
- Do NOT translate entities

RETURN ONLY THE FINAL JSON RESULT (no explanations, no markdown):
{"person": [], "organizations": [], "address": []}
```

**Result:**

```json
{
  "person": [
    "Lương Văn Tuấn",
    "Nguyễn Xuân Nam",
    "Lô Thị Hoa",
    "Kha Thị Tím",
    "Nguyễn Mạnh Lợi",
    "Anh-đô Kát-su-hi-rô"
  ],
  "organizations": [
    "Vườn quốc gia Pù Mát",
    "Trường tiểu học 3 Môn Sơn",
    "Trường tiểu học Môn Sơn",
    "Hợp tác xã thủ công mỹ nghệ xã Môn Sơn",
    "Cơ quan hợp tác quốc tế Nhật Bản (JICA)",
    "Viện Nghiên cứu ngành nghề nông nghiệp Việt Nam",
    "Sở Du lịch Nghệ An",
    "UBND huyện Con Cuông",
    "Ban quản lý Vườn quốc gia Pù Mát",
    "UNESCO",
    "Khu dự trữ sinh quyển thế giới"
  ],
  "address": [
    "miền tây xứ Nghệ",
    "huyện Con Cuông",
    "Nghệ An",
    "Vườn quốc gia Pù Mát",
    "xã Môn Sơn",
    "xã biên giới giáp nước bạn Lào",
    "đập nước Phà Lài",
    "trung tâm thị trấn Con Cuông",
    "quận thể Vườn quốc gia Pù Mát",
    "huyện Anh Sơn",
    "huyện Tương Dương",
    "bản Cò Phạt",
    "bản Nưa",
    "xã Yên Khê",
    "bản Xiềng",
    "bản Pha",
    "điểm trường bản Cò Phạt",
    "điểm giới thiệu, bán sản phẩm của hợp tác xã",
    "đường liên thôn, liên xã",
    "thác Khe Kèm",
    "khe nước Mọc",
    "hang Thẳm Ồm",
    "di tích lịch sử, văn hóa bia Ma Nhai",
    "thành cổ Trà Lân",
    "homestay Hoa Thụ",
    "Việt Nam"
  ]
}
```

### 2.1.2. Experimental Results

#### Overall Performance Comparison

| Method | Exact Match | Overall F1 | Precision | Recall | Avg Time/Example |
|--------|-------------|------------|-----------|--------|------------------|
| **Few-Shot** | 16.0% | 33.8% | 55.6% | 24.3% | 2.81s |
| **Zero-Shot** | 16.0% | 27.1% | 39.4% | 20.7% | 2.18s |
| **Chain-of-Thought** | 18.0% | 33.3% | 41.3% | 27.9% | 7.21s |

#### Per-Entity-Type Performance (F1-Score)

| Method | Person | Organization | Address |
|--------|--------|--------------|---------|
| **Few-Shot** | 35.1% | 21.3% | 38.3% |
| **Zero-Shot** | 30.9% | 18.9% | 27.0% |
| **Chain-of-Thought** | 41.8% | 25.6% | 19.9% |


## 2.2 Retrival Augmented Generation (RAG)

**Dataset:** `dev`

**Model:** `mistral:7b`

### 2.2.1. Chunking

Using Vietnamese sentence segmentation library `underthesea`

**Chunking configuration:**


```python
CONFIG = {
    "chunk_size": 1500,           # Maximum characters per chunk
    "chunk_overlap": 100,          # Overlap between chunks (characters)
    "min_chunk_size": 50,          # Minimum chunk size to keep
}
```

**Chunking  process:**


```python
def chunk_with_metadata(text: str, doc_id: str, metadata: Dict) -> List[Dict]:
    sentences = sent_tokenize(text)  # Vietnamese sentence segmentation
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        if current_length + len(sent) <= chunk_size:
            current_chunk.append(sent)
            current_length += len(sent)
        else:
            # Save chunk
            chunks.append({
                "text": " ".join(current_chunk),
                "doc_id": doc_id,
                "chunk_index": len(chunks),
                "char_count": current_length,
                "metadata": metadata
            })

            # Start new chunk with overlap (last 2 sentences)
            overlap_sents = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
            current_chunk = overlap_sents + [sent]
            current_length = sum(len(s) for s in current_chunk)

    return chunks

```

**Vector DB:** `Chroma DB`

**Embedding model:** `nomic-embed-text:latest` via Ollama  (768 dimensions)

### 2.2.2. Retrieval

**Retrieval Pipeline:**

1. **Query Embedding**: Convert input text to 768-dimensional vector
2. **Similarity Search**: Find top-k most similar chunks from knowledge base
3. **Deduplication**: Remove redundant chunks from same document (max 1 chunk per document)
4. **Context Building**: Construct RAG prompt with retrieved examples



**LLM Query:**

```txt
You are an expert Vietnamese Named Entity Recognition (NER) system.

Task: Extract named entities from Vietnamese text into three categories.
[... detailed instructions ...]

### Similar Examples from Knowledge Base:

Example 1:
Text: Dạy học cho học sinh vùng dân tộc đã khó, nhưng để dạy được tiếng Anh...
[Full example text and groundtruth from retrieved chunk]

Example 2:
Text: Để dựa vào đó, các vị giám khảo có thể đánh giá một cách công tâm...
[Full example text and groundtruth from retrieved article]

[... more examples ...]

## Input Text:
"""
[New text to process]
"""

## Output (JSON only, no explanation):
```

### 2.2.3. Result

#### Overall Performance Comparison

| Method | Exact Match | Overall F1 | Precision | Recall | Avg Time/Example |
|--------|-------------|------------|-----------|--------|------------------|
| **Baseline (No RAG)** | 10.0% | 45.8% | 49.6% | 24.3% | 2.81s |
| **With RAG** | 40.0% | 63.1% | 53.4% | 51.7% | 2.18s |

#### Per-Entity-Type Performance (F1-Score)

| Method | Person | Organization | Address |
|--------|--------|--------------|---------|
| **Baseline (No RAG)** | 45.1% | 47.3% | 38.3% |
| **With RAG** | 65.9% | 61.6% | 27.0% |


## 2.3 Finetune Model

**Dataset:** `dev`

**Model:** `mistral:7b`

Apply PEFT (Parameter Efficient Fine-Tuning) LoRA to improve NER task of model

### 2.3.1. Perpare dataset

**Dataset:**



| Dataset | Number of samples | Percent (%) |
|---------|-------------------|-------------|
|  Train | 234 | 90 |
| Val | 234 | 10 |

**Data format:**

```json
### Instruction:
You are a Vietnamese Named Entity Recognition (NER) expert. Extract named entities
from the given text and classify them into three categories:
- person: Names of people
- organizations: Names of organizations, companies, institutions
- address: Location names, addresses

Return your answer as a JSON object with these three keys. Each value should be a
list of strings. If a category has no entities, return an empty list. Do not invent
entities that are not present in the text.

### Input:
[Vietnamese text to analyze]

### Response:
{
  "person": ["Entity 1", "Entity 2"],
  "organizations": ["Entity 1"],
  "address": ["Entity 1", "Entity 2", "Entity 3"]
}
```

### 2.3.2. Train Model

**Train configuration:**


```python
TrainingArguments(
    # Optimization
    optim="paged_adamw_8bit",           # 8-bit Adam for memory efficiency
    learning_rate=2e-4,                  # Standard for QLoRA
    weight_decay=0.01,                   # L2 regularization
    lr_scheduler_type="cosine",          # Cosine annealing
    warmup_ratio=0.03,                   # 3% warmup steps
    max_grad_norm=1.0,                   # Gradient clipping

    # Precision
    bf16=True,                           # BFloat16 training
    fp16=False,

    # Batch configuration
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,       # Effective batch size = 8
    eval_accumulation_steps=8,
)
```

**Hardware Configuration:**

- **CPU**: Intel i5 13500

- **Memory**: 64Gb

- **GPU**: NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)

- **Data workers**: 2 parallel workers


**Train chart:**

![train_chart](./docs/imgs/train.png)

**Checkpoint Structure:**

```
checkpoints/mistral-7B-Instruct-v0.3-ner-qlora/
├── final-adapters/              # Final LoRA adapters
│   ├── adapter_config.json      # LoRA configuration
│   ├── adapter_model.bin        # LoRA weights (~675 MB)
│   └── tokenizer files          # Tokenizer config
├── checkpoint-100/              # Intermediate checkpoint
├── checkpoint-200/              # (only saves last 3)
├── logs/                        # TensorBoard logs
└── trainer_state.json           # Training state
```

**Adapter Size:**

- **LoRA adapters only**: 675.56 MB
- **Storage efficiency**: Only adapters needed for deployment


### 2.3.3. Result

#### Overall Performance Comparison

| Method                  | Exact Match | Overall F1 | Precision | Recall | Avg Time/Example |
|-------------------------|-------------|------------|-----------|--------|------------------|
| **Base model**          | 3.3%        | 33.3%      | 66.2%     | 22.3%  | 2.81s            |
| **Fine-tuned (QLoRA)**  | 6.7%        | 47.5%      | 75.3%     | 34.7%  | 2.18s            |

#### Per-Entity-Type F1-Scores

| Method                  | Person | Organization | Address |
|-------------------------|--------|--------------|---------|
| **Base model**          | 24.6%  | 43.5%        | 14.8%   |
| **Fine-tuned (QLoRA)**  | 54.3%  | 40.6%        | 55.1%   |





