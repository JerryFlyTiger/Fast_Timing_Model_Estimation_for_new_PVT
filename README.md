# Fast_Timing_Model_Estimation_for_new_PVT
## 2020 cad contest problem D  
 [點我連官網](http://iccad-contest.org/2020/tw/problems.html)  

## 題目簡介  

### Introduction  
    
        在 ASIC (Application-Specific Integrated Circuit) 及 SoC (System on Chip) 設計流程中 standard cell library  
	扮演著十分重要的角色，從模擬、合成到物理實體佈局都需要透過 standard cell library 來取得 cell 相關資訊，  
	例如:timing、power、area......等資訊，並利用這些資訊來完成晶片設計。  
	    為了確保工程師所設計的電路與實際生產出來的晶片行為一致，因此如何讓 standard cell library 具有高準確度是非常  
	重要的課題。產生 standard cell library 的流程稱為 library characterization，現行的流程需要使用 SPICE model、SPICE  
	netlist 及 characterization tool 並利用窮舉法來產生所有結果，此過程非常耗時 且耗費運算資源。  
	    在先進製程中，電晶體特性模型越來越複雜，更是加劇了這個缺點。根據設計規格，設計過程中 standard cell library  
	可能需要新的 PVT(Process, Voltage, Temperature) corners，但 characterization 過程耗時以及運算資源有限，若能先  
	characterize 出 少數 cells，經由 machine learning 方法快速得到一版 cell timing，將有助於設計時程加速以及不同  
	PVT 下的 design performance 快速探索與評估。  
  
### Problem Description  
  
        參賽者必須使用機器學習方法來訓練出準確的預測模型，例如:Tensorflow、PyTorch、 MATLAB，並將模型與輸出的  
	library 一同上傳繳交。本次題目使用的 library 共包含 700 個 cell。PVT condition 共 15 種，各抽取 400 個 cell  
	作為 training set 供參賽者訓練模型使用，並於競賽各階段提供 inference 資訊及模板， 其內容包含 training set  
	以外的 100 個新 cell。本次競賽提供的模板為 timing table、power table 挖空後的框架。參賽者必須利用訓練出來的  
	模型推導出相同 100 個 cell 的其他 PVT 的 timing 及 power 資訊再填入模板。  
  
### 程式流程：  

要做到能預測沒看過的 cell 的其他 PVT 的 timing 及 power 資訊  

![program flow](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/programFlow.jpg)  

### 訓練流程：  

用 280 個 cell 當訓練資料，剩餘 120 個拿來測試  

![training flow](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/trainingFlow.jpg)  

### 前處理流程：  

因為不是 csv 檔，所以要先處理  

![preprocess flow](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/preprocess.jpg)  

### 範例資料  

此資料檔只是一小部分，全部總共有 15 個檔案，也就是 15 種 PVT condition  

[點我連 data](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/tree/master/data)  

## todo list

- [x] 規劃
- [ ] 前處理
  - [ ] 寫文字處理程式將 cell library 轉成 csv 檔
  - [ ] 分七成為訓練資料，三成為測試資料
- [ ] 訓練
  - [ ] 撰寫機器學習程式
  - [ ] 以題目指定的評估精準度公式撰寫評估程式
- [ ] 評估

## 進度報告書

[倉庫內的進度報告的連結](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/%E9%80%B2%E5%BA%A6%E5%A0%B1%E5%91%8A/%E9%80%B2%E5%BA%A6%E5%A0%B1%E5%91%8A.pdf)

## Problem D 更多資訊  

[倉庫內的 pdf 檔的連結](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/ProblemD_0201.pdf)  

## 競賽海報：  
![海報](https://github.com/JerryFlyTiger/Fast_Timing_Model_Estimation_for_new_PVT/blob/master/2020CAD_%E5%9C%8B%E5%85%A7%E8%B3%BD%E5%AE%9A%E7%A8%BF%E6%B5%B7%E5%A0%B1.jpg)  

