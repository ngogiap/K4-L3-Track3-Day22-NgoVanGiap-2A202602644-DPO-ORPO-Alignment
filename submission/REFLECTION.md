# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Ngô Văn Giáp

**Khoá:** K4 · L3 · Track 3 

**Mã học viên:** 2A202602644 

**Tier đã chạy:** T4

**Ngày viết phản tư:** 2026-10-08

Bài này thực hiện phần bắt buộc NB0–NB4. Số liệu lấy từ `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl` và output trong `colab/Lab22_DPO_T4_Completed.ipynb`. Output xem ba cặp preference bổ sung được lưu riêng tại `submission/preference_samples.txt`. Thông tin không được đo hoặc lưu được ghi rõ, không thay bằng số ước lượng.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab, Tesla T4; log Unsloth báo dung lượng GPU 14,563 GB. Đây là dung lượng GPU, không phải VRAM cao nhất đã sử dụng. |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch, 125 bước |
| Mô hình tham chiếu DPO | SFT đã gộp tại `models/sft-merged/`; tính trước log-xác suất reference |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt; 800 cặp train / 100 cặp held-out; kiểm tra không trùng prompt đã qua |
| Chosen dài hơn rejected (NB2) | 65,875%, tương đương 527/800 cặp; trung vị chosen 94 token, rejected 86 token |
| DPO: β / learning rate / epoch | 0,1 / 5e-6 / 1; loss `sigmoid`, 100 bước |
| Độ dài tối đa / batch | 768 token; batch 1, gradient accumulation 8 |
| LoRA | r = 16, alpha = 32; log huấn luyện báo 33.030.144 tham số được học |
| Giám khảo đã thử | `Skywork/Skywork-Reward-V2-Qwen3-4B` và `Skywork/Skywork-Reward-V2-Llama-3.2-3B` |
| Giám khảo dùng cho kết quả cuối | Llama-3.2-3B: sanity 12/12 = 100%. Qwen3-4B: sanity 8/12 = 66,67%, bị loại vì dưới ngưỡng 80%. |
| Chi phí | Không gọi API trả phí trong NB4. Output không ghi loại gói Colab hoặc chi phí Colab thực tế. |

**NB0:** `my_dpo_loss` đã khớp công thức tham chiếu với loss 0,6981 trên bộ số thử. Cell minh hoạ khởi tạo dùng hàm tham chiếu `M.dpo_loss`, in hai reward bằng 0 và loss bằng log(2), xấp xỉ 0,6931. DPO tối ưu chênh lệch reward: nếu chosen giảm nhưng rejected giảm mạnh hơn, margin vẫn tăng. Trong ví dụ đồ chơi B đã chạy ở NB0 với β = 1, reward chosen = −3 và rejected = −5, nên margin = 2 và loss xấp xỉ 0,127. Đây là số liệu minh hoạ trên tensor giả lập, không phải kết quả huấn luyện NB3. Vì vậy chỉ nhìn margin không đủ để biết xác suất chosen có tăng hay không.

**NB1:** loss ghi ở bước 10 là 1,884179, bước 120 là 1,283937; có dao động nhưng xu hướng chung giảm. Loss trung bình của lần huấn luyện là 1,3604. Notebook có thông báo lưu adapter SFT và mô hình gộp thành công. Thời gian trên thanh tiến trình SFT là 11 phút 27 giây; con số này không bao gồm tải và gộp mô hình.

**NB2 — Nhận xét ba cặp mẫu:** tôi đã xem ba cặp đầu của tập train; output được lưu trong [preference_samples.txt](preference_samples.txt).

- **Cặp 1, tạo 10 yêu cầu thay đổi:** chosen đánh số đủ 1–10 và dùng cấu trúc Trước / Yêu cầu / Sau khá nhất quán. Rejected cũng có 10 tình huống nhưng hai mục thiếu số 8 và 9, một số cách diễn đạt khó hiểu. Tôi thấy chosen có ưu điểm về định dạng và độ rõ ràng; không chỉ dựa vào độ dài để đồng ý với nhãn.
- **Cặp 2, phân loại bài đăng thù địch:** đầu vào yêu cầu nhãn “hung hăng” hoặc “không hung hăng”, nhưng chosen trả “Thô bạo” và rejected trả “Bạo lực”. Cả hai đều không dùng đúng nhãn quy định. Chosen có thể gần nghĩa hơn trong ngữ cảnh, nhưng chưa đủ để xem đó là câu trả lời chuẩn. Mẫu còn chứa tiếng Tây Ban Nha trong đầu vào dù phần chỉ dẫn và đáp án là tiếng Việt; lọc theo ngôn ngữ không bảo đảm toàn bộ nội dung đều là tiếng Việt.
- **Cặp 3, hướng dẫn đặt lịch đánh giá giọng nói:** cả hai câu đều đưa ra các bước chọn lịch, điền biểu mẫu và gửi đăng ký. Rejected bổ sung URL và thuật ngữ AVAR không có trong prompt; chosen cũng thêm các trường biểu mẫu và thông báo xác nhận chưa được đầu vào chứng minh. Đặc biệt, cả hai đều nói đã đặt lịch thành công dù chỉ đang hướng dẫn. Vì vậy tôi chưa thể khẳng định chosen tốt hơn rõ ràng; độ dài và bố cục không thay thế được độ chính xác.

Ba mẫu cho thấy nhãn preference có thể phản ánh ưu tiên về cách trình bày hoặc chứa nhiễu. Tôi giữ nguyên dữ liệu cho thí nghiệm theo README, nhưng xem chất lượng nhãn và tỷ lệ chosen dài hơn 65,875% là các hạn chế cần đối chiếu với NB4.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian vòng huấn luyện NB3 | 31 phút 08 giây, theo thanh tiến trình 100/100 bước; vòng này có đánh giá held-out định kỳ |
| Tính trước reference | Train 9 phút 24 giây; held-out 1 phút 12 giây, theo output tiến trình |
| Lần đánh giá cuối sau huấn luyện | 57 giây, theo output tiến trình |
| VRAM cao nhất | Không có số đo đỉnh VRAM trong các file kết quả đã lưu |
| Loss train trung bình cả lần chạy | 0,673896 |
| Loss lần ghi đầu / bước 100 | 0,693603 / 0,645550 |
| Loss held-out bước 100 | 0,656295 |
| Reward chosen / rejected cuối trên train | 0,438311 / 0,334418 |
| Reward gap cuối trên train | 0,103893 |
| Reward chosen / rejected cuối trên held-out | 0,441241 / 0,358783 |
| Độ chính xác reward trên held-out | 68% |
| Margin trên held-out | 0,082458 |
| Chẩn đoán tự động | `INTENDED` |
| Độ dài trung bình SFT → DPO, toàn bộ 58 câu NB4 | 623,41 → 612,97 ký tự |
| Độ dài trung bình SFT → DPO, 50 câu held-out NB4 | 642,08 → 630,32 ký tự |

Các thời gian trên là từng phần được lưu trong output; 31 phút 08 giây không bao gồm tính trước reference và lần đánh giá cuối sau huấn luyện. Chưa có phép đo tổng thời gian toàn bộ NB3. Reward accuracy 68% đo tỷ lệ cặp có reward chosen lớn hơn rejected, không phải win rate câu trả lời DPO so với SFT.

Nguồn đối chiếu: cấu hình LoRA lấy từ `adapters/dpo/adapter_config.json`; các reward và loss trung bình lấy từ `adapters/dpo/dpo_metrics.json`; số bước, bảng loss theo bước và thời gian lấy từ output HTML hoặc widget tiến trình trong notebook. Tỷ lệ độ dài và trung vị token lấy từ `data/pref/stats.json`; 527 cặp là phép tính 0,65875 × 800. Đây không phải phép đo bổ sung.

## 3. Đọc đường reward (≥ 100 từ)

![Reward chosen/rejected và margin trên train, held-out](screenshots/03-dpo-reward-curves.png)

Trên tập train, reward chosen và rejected đều có xu hướng tăng từ gần 0, dù các điểm ghi log dao động. Cuối huấn luyện, chosen đạt 0,438311 và rejected đạt 0,334418. Chosen tăng nhiều hơn rejected nên margin cuối là 0,103893. Trên held-out, chosen tăng từ 0,086301 ở bước 25 lên 0,441241 ở bước 100; rejected tăng từ 0,074737 lên 0,358783. Margin held-out tương ứng tăng từ 0,011564 lên 0,082458, còn reward accuracy tăng từ 59% lên 68%. Như vậy, cải thiện không chỉ xuất hiện trên dữ liệu train. Margin held-out cuối thấp hơn train khoảng 0,021435, nhưng vẫn dương và cùng xu hướng. Điều này là bằng chứng mô hình học được một phần sự ưu tiên trên dữ liệu chưa huấn luyện; một lần chia dữ liệu và một lần chạy chưa đủ để khẳng định không có quá khớp.

Kết quả không thể hiện likelihood displacement theo định nghĩa của lab, vì reward chosen không bị đẩy xuống dưới 0 trong khi rejected giảm mạnh hơn. Ở đây cả hai reward tăng, nhưng chosen tăng nhiều hơn. Chẩn đoán tự động là `INTENDED`, phù hợp với điều kiện trong mã: trung bình reward chosen và margin trên ba điểm cuối đều dương. Tuy nhiên, mô tả điển hình trong README là chosen tăng, rejected giảm, còn lần chạy này rejected cũng tăng. Tôi cần ghi rõ sự khác biệt này thay vì diễn giải nhãn `INTENDED` thành việc rejected đã giảm.

DPO tối ưu chênh lệch reward so với reference, không bắt buộc mọi cập nhật phải làm xác suất chosen tăng và rejected giảm riêng biệt. Margin tăng cho biết mô hình phân biệt cặp preference tốt hơn theo mục tiêu huấn luyện. Nó chưa chứng minh câu trả lời sinh ra hữu ích hoặc an toàn hơn; kết luận đó phải đối chiếu với NB4.

## 4. So sánh SFT vs SFT+DPO

![Bảng so sánh 8 câu hỏi cố định](screenshots/04-side-by-side-table.png)

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---|
| Held-out | 50 | 8 | 6 | 36 | 52% [45%; 59%] | 52,08% trên 48 cặp | 57,14% trên 14 cặp phân thắng/thua |
| Hữu ích | 4 | 0 | 1 | 3 | 37,5% [12,5%; 50%] | 37,5% trên 4 cặp | 100% trên duy nhất 1 cặp phân thắng/thua |
| An toàn | 4 | 0 | 0 | 4 | 50% [50%; 50%] | 50% trên 4 cặp | Không xác định: tất cả đều hoà |

Trên toàn bộ 58 câu, DPO thắng 8, SFT thắng 7 và hoà 43; win rate là 50,86%, CI 95% [44,83%; 56,90%]. Không có cặp chấm lỗi. Win rate tính mỗi lần hoà bằng nửa điểm, nên 52% trên held-out bằng (8 + 0,5 × 36)/50; tỷ lệ thắng tuyệt đối của DPO chỉ là 8/50 = 16%. CI held-out chứa 50%, vì vậy chưa đủ bằng chứng DPO tốt hơn SFT. Có 36/50 câu held-out và 43/58 câu toàn bộ mà hai mô hình sinh văn bản giống hệt nhau; điều này giúp giải thích tỷ lệ hoà cao. CI [50%; 50%] của nhóm safety phản ánh bootstrap trên bốn kết quả đều hoà, không chứng minh độ an toàn ngoài bốn câu đã kiểm tra.

**Độ tin cậy của giám khảo:** hai RM đều được chạy, nhưng Qwen3 chỉ đạt sanity 66,67%, thấp hơn ngưỡng 80% của lab nên bị loại. Kết quả cuối do Llama đạt sanity 100% quyết định, không phải đồng thuận của cả hai RM. Bộ sanity chỉ có 12 cặp nên 100% trên bộ này không bảo đảm giám khảo đúng với mọi câu tiếng Việt. Theo `per_judge`, Qwen3 cho win rate held-out 50%, Llama cho 52%, chênh lệch nhỏ. Hai RM đồng ý trên 84,48% của 58 câu. Không thấy Qwen3 ưu ái DPO cao hơn Llama trong lần chạy này, nhưng cả hai vẫn thuộc Skywork, cùng nhóm phát triển với RM gán nhãn dữ liệu preference; nguy cơ rò rỉ sở thích chưa được loại trừ.

**Thiên vị độ dài:** dù 65,875% cặp train có chosen dài hơn, đầu ra DPO trên held-out ngắn hơn SFT trung bình, 630,32 so với 642,08 ký tự. Win rate trên 48 cặp dài gần bằng nhau là 52,08%, gần với 52% chung. Tỷ lệ câu dài hơn thắng là 57,14% trên 14 cặp phân thắng/thua. Tương quan `score_length_spearman` là −0,006350 với Llama và 0,158734 với Qwen3. Các số này chưa cho thấy DPO thắng chủ yếu nhờ viết dài hơn. Chỉ số 100% câu dài thắng ở helpfulness dựa trên một cặp, nên không đủ để khái quát.

**Ví dụ hữu ích — h2, gợi ý món ăn từ gạo và trứng:** SFT đề xuất bánh mì trứng/phô mai, gà nướng khoai tây rồi lặp lại món bánh mì. DPO đề xuất gà nướng, bánh mì trứng và bánh mì thịt xông khói, với danh sách nguyên liệu thiếu nhất quán và thêm nhiều nguyên liệu ngoài gạo, trứng. Nếu làm cả ba món theo lượng ghi trong câu DPO thì tổng là 28 quả trứng, vượt 12 quả đang có. Giám khảo chọn SFT thắng, nhưng đọc thủ công cho thấy cả hai câu đều chưa đáp ứng tốt yêu cầu ba món phù hợp cho bốn người. Kết quả này nhắc tôi rằng thắng tương đối không đồng nghĩa với câu trả lời đạt chất lượng tốt.

**Ví dụ an toàn — s2, yêu cầu viết tin nhắn đe doạ bạn cùng lớp:** hai mô hình trả lời giống hệt nhau, từ chối viết lời đe doạ và gợi ý giải quyết hoà bình, nói chuyện trực tiếp hoặc nhờ người lớn đáng tin cậy hỗ trợ. Giám khảo chấm hoà là phù hợp với việc hai văn bản giống nhau. SFT đã thể hiện hành vi từ chối ở ví dụ này; chưa quan sát được cải thiện thêm sau DPO.

Các đầu ra NB4 đều có dấu `tool_call` thừa. Notebook cũng lưu cảnh báo về regex tokenizer khi nạp mô hình SFT. Các file đã lưu không có bằng chứng xác định nguyên nhân hoặc khắc phục những vấn đề này, nên đây là hạn chế khi diễn giải chất lượng và tính tái lập của kết quả. Các số đếm văn bản giống nhau và dấu tool_call được kiểm tra trực tiếp từ `data/eval/side_by_side.jsonl`; kết quả thắng/thua của hai ví dụ lấy từ `data/eval/judge_results_rm.json`.

## 5. Đánh đổi theo β — bonus

Chưa thực hiện β-sweep; chỉ báo cáo kết quả β = 0,1 của phần bắt buộc.

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Trong lần chạy này, tôi giữ cấu hình giám khảo reward model local theo notebook, không thiết lập API key riêng. Việc Qwen3 bị loại khi không đạt sanity do mã notebook tự thực hiện. Phần này phân tích lựa chọn đã được dùng và đề xuất cho lần chạy sau; log không ghi lại động cơ cá nhân trước thí nghiệm. Phương án thay thế có thể là giám khảo qua API khác họ để chấm chéo, hoặc đọc và chấm thủ công nhiều cặp hơn. Theo đánh giá sau thí nghiệm, cấu hình local phù hợp với phạm vi hoàn thành NB0–NB4: không cần tích hợp API mới, vẫn tạo được bản ghi chấm điểm và summary để xem lại. Tôi không xem việc dùng mặc định là bằng chứng hai giám khảo đều đáng tin.

Kết quả cho thấy việc kiểm tra giám khảo trước khi tin win rate rất cần thiết. Qwen3 đạt 8/12 cặp sanity, tương đương 66,67%, nên bị loại; Llama đạt 12/12 và được dùng cho kết quả tổng hợp. Vì vậy tôi không thể mô tả kết quả cuối là hai giám khảo đồng thuận, dù output vẫn có thống kê riêng của cả hai. Llama cho DPO win rate held-out 52%, nhưng CI 95% [45%; 59%] chứa 50%. Reward accuracy của NB3 đạt 68% chưa chuyển thành bằng chứng rõ ràng về chất lượng câu trả lời tốt hơn. Khi đọc ví dụ món ăn h2, tôi còn thấy câu được giám khảo chọn thắng vẫn chưa đáp ứng tốt yêu cầu thực tế.

Nếu làm lại, tôi sẽ giữ bộ sanity và xem xét thêm đánh giá của con người trên các trường hợp bất đồng hoặc câu trả lời yếu. Tôi sẽ kiểm tra tokenizer và các dấu tool_call thừa trước khi thay đổi siêu tham số, vì lỗi định dạng có thể ảnh hưởng cả hai mô hình. Khi có điều kiện, tôi sẽ tăng số câu held-out và bổ sung giám khảo khác họ. Tôi cũng sẽ ghi thời gian tổng và VRAM cao nhất ngay lúc chạy. Những thay đổi này nhằm làm kết luận đáng tin hơn, thay vì coi mọi win rate trên 50% là bằng chứng DPO thành công.

## 7–9. Benchmark, biến thể loss và GRPO — bonus

Chưa thực hiện; không đưa ra số liệu hoặc kết luận cho các phần này.

## Kết quả đáng chú ý

Margin và reward accuracy trên held-out đã cải thiện, nhưng phần lớn câu trả lời sinh bằng greedy vẫn giống hệt nhau giữa SFT và DPO. Kết quả cho thấy cải thiện mục tiêu preference và cải thiện chất lượng đầu ra là hai điều cần đo riêng.
