# Fast_Timing_Model_Estimation_for_new_PVT
 2020 cad contest problem D  
 [2020 cad contest 官網](http://iccad-contest.org/2020/tw/problems.html)  
 I. Introduction  
    
        在 ASIC (Application-Specific Integrated Circuit) 及 SoC (System on Chip) 設計流程中 standard cell library  
	扮演著十分重要的角色，從模擬、合成到物理實體佈局都需要透過 standard cell library 來取得 cell 相關資訊，例如:timing、  
	power、area......等資訊，並利用這些資訊來完成晶片設計。為了確保工程師所設計的電路與實際生產出來的晶片行為一致，因此，  
	如何讓 standard cell library 具有高準確度是非常重要的課題。產生 standard cell library 的流程稱為 library characterization，  
	現行的流程需要使用 SPICE model、SPICE netlist 及 characterization tool 並利用窮舉法來產生所有結果，此過程非常耗時 且耗費運算資源。  
	在先進製程中，電晶體特性模型越來越複雜，更是加劇了這個缺點。根據設計規格，設計過程中 standard cell library 可能需要新的 PVT  
	(Process, Voltage, Temperature) corners，但 characterization 過程耗時以及運算資源有限，若能先 characterize 出 少數 cells，  
	經由 machine learning 方法快速得到一版 cell timing，將有助於設計時程加速以及不同 PVT 下的 design performance 快速探索與評估。  
  
II. Problem Description  
  
        參賽者必須使用機器學習方法來訓練出準確的預測模型，例如:Tensorflow、PyTorch、 MATLAB，並將模型與輸出的 library 一同上傳繳交。  
	本次題目使用的 library 共包含 700 個 cell。PVT condition 共 15 種，各抽取 400 個 cell 作為 training set 供參賽者訓練模型使用，  
	並於競賽各階段提供 inference 資訊及模板， 其內容包含 training set 以外的 100 個新 cell。本次競賽提供的模板為 timing table、  
	power table 挖空後的框架。參賽者必須利用訓練出來的模型推導出相同 100 個 cell 的其他 PVT 的的 timing 及 power 資訊再填入模板。  
  
程式流程：  
![programFlow]("/Users/jerrychen/OneDrive/文件/cadContest/R_version/Fast_Timing_Model_Estimation_for_new_PVT")  

