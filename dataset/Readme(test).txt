檔名：test_info.csv

unique_id：選手揮拍測驗的id(對應test_data檔名)
mode：預測揮拍模式(參照「AI CUP 2025桌球智慧球拍資料的精準分析-程式介紹.ppt」第六頁說明)
cut_point：每次揮拍的資料切割節點，僅供參考

=====================================================================================
檔名：sample_submission.csv

unique_id：選手揮拍測驗的id(對應test_data檔名)
gender : 預測男生機率
hold racket handed : 預測右手機率
play years_0 : 預測低球齡機率
play years_1 : 預測中球齡機率
play years_2 : 預測高球齡機率
level_2 : 預測大專甲組選手機率
level_3 : 預測大專乙組選手機率
level_4 : 預測青少年國手機率
level_5 : 預測青少年選手機率


=====================================================================================
資料夾：test_data

資料夾中每個txt檔為選手的揮拍數據紀錄：
	1.紀錄選手一次測驗中揮拍的連續數據
	2.一次測驗包含27次揮拍，包含測驗前後的擾動資訊

=====================================================================================
檔名：*.txt

X軸加速度(Ax)，Y軸加速度(Ay)，Z軸加速度(Az)，X軸角速度(Gx)，Y軸角速度(Gy)，Z軸角速度(Gz)

