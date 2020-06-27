rm( list = ls() )

if ( !require( readr ) ) {
  install.packages( "readr" )
  library( readr )
} # if

# 讀取 cell library 資料
statement_std  <- read_file( './base_nom_0p9v/lib1_ff0p99v125c_base_400.tlib' )
statement_drop <- read_file( './base_nom_0p8v/lib1_ff0p88v125c_base_400.tlib' )

# 設定 timing power 的 regular expression
timingPattern <- "(timing [(][)] ([{]((?>[^{}]|(?2))*)[}]))"
powerPattern  <- "(internal_power [(][)] ([{]((?>[^{}]|(?2))*)[}]))"

# 取得 cell library 中的 values
getValues <- function( pattern, statement ) {
  # 以 gregexpr 搜尋符合 pattern 的子字串，然後以 substring 擷取該子字串
  # 以下兩行擷取到 timing (){...}
  matched <- gregexpr( pattern, statement, perl = T )
  tmpStr  <- substring( statement, matched[[1]], matched[[1]] + attr( matched[[1]], "match.length" ) - 1 )
  values <- c()
  result <- sapply( tmpStr, function( ts ) {
    # 以下兩行從 timing (){...} 擷取到 value (...)
    matched   <- gregexpr( "(values [(][^)]+[)];)", ts, perl = T )
    valueStrs <- substring( ts, matched[[1]], matched[[1]] + attr( matched[[1]], "match.length" ) - 1 )
    sapply( valueStrs, function( vals ) {
      # 以下兩行從 value (...) 擷取到 數字
      matched <- gregexpr( "[-]?[[:digit:]]+[.]?[[:digit:]]*(e-[[:digit:]]+)?", vals, perl = T )
      numTmp  <- as.numeric( substring( vals, matched[[1]], matched[[1]] + attr( matched[[1]], "match.length" ) - 1 ) )
      if ( length( numTmp ) != 49 ) {
        print(numTmp )
        exit()
      }
      
      if ( numTmp[1] != 0  )
        values <<- c( values, numTmp )
    })
    
    return( t )
  })
  
  return( values )
} # getValues

std_tVal <- getValues( timingPattern, statement_std )
std_pVal <- getValues( powerPattern, statement_std )

drop_tVal <- getValues( timingPattern, statement_drop )
drop_pVal <- getValues( powerPattern, statement_drop )

# 建立模型並驗證
validation <- function( y, x, validation_y, validation_x, i, l ) {
  relation <- lm( y~x )
  
  # Give the chart file a name.
  png( file = paste( "linearRegression", l, i, ".png", sep="" ) )
  
  # Plot the chart. 視覺化展示 x, y 間的關係
  plot( y, x, col = "blue", main = "std & drop Regression",
        abline( lm( x~y ) ), xlab = "std", ylab = "drop" )
  
  # Save the file.
  dev.off()
  
  # print( relation )
  # print( summary( relation ) )
  predictions <- predict( relation, data.frame( x = validation_x ) )
  tmp <- abs( validation_y - predictions ) / abs( validation_y )
  tmp[tmp > 1] = 1
  return( 100 - 100 * mean((sqrt(tmp**2))) )
} # validation

# 重整資料：將資料切成十份
splitTo10Data <- function( dataSet ) { return( split( dataSet, ceiling( seq_along( dataSet ) / ( length( dataSet ) / 10 ) ) ) ) }

std_tVal <- splitTo10Data( std_tVal )
std_pVal <- splitTo10Data( std_pVal )
drop_tVal <- splitTo10Data( drop_tVal )
drop_pVal <- splitTo10Data( drop_pVal )

valid_10_data <- function( y_Val, x_Val, l ) {
  sapply( 1 : 10, function( i ) {
    # 取一份出來當驗證數據，其他為訓練數據
    drop_trainingSet_tVal <- unlist( y_Val[-i], use.names=FALSE )
    std_trainingSet_tVal  <- unlist( x_Val[-i], use.names=FALSE )
    drop_validationSet_tVal <- unlist( y_Val[i], use.names=FALSE )
    std_validationSet_tVal  <- unlist( x_Val[i], use.names=FALSE )
    score <- validation( drop_trainingSet_tVal, std_trainingSet_tVal, drop_validationSet_tVal, std_validationSet_tVal, i, l )
    print( score )
  })
} # valid_10_data

valid_10_data( drop_tVal, std_tVal, 'Timing' )
valid_10_data( drop_pVal, std_pVal, 'Power' )
