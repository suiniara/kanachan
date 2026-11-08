param(
    [string]$InputReceipt='runs\remote-data\decoder-bc-summary.json',
    [string]$Plan='runs\remote-data\decoder-bc-plan.json',
    [string]$Output='runs\remote-data\decoder-bc-learning.png',
    [ValidateSet('bc','cql')][string]$Kind='bc'
)
$ErrorActionPreference='Stop'
if($Kind -eq 'cql'){
    if(-not $PSBoundParameters.ContainsKey('InputReceipt')){$InputReceipt='runs\remote-data\decoder-cql-summary.json'}
    if(-not $PSBoundParameters.ContainsKey('Plan')){$Plan='runs\remote-data\decoder-cql-plan.json'}
    if(-not $PSBoundParameters.ContainsKey('Output')){$Output='runs\remote-data\decoder-cql-learning.png'}
}
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$taskReceipt=Get-Content -LiteralPath $InputReceipt -Raw -Encoding UTF8|ConvertFrom-Json
$taskPlan=Get-Content -LiteralPath $Plan -Raw -Encoding UTF8|ConvertFrom-Json
if($null -ne $taskReceipt.status -and $taskReceipt.status -isnot [string]){$taskReceipt=$taskReceipt.status}
if($Kind -eq 'cql'){
    $taskMetrics=Join-Path (Split-Path ([IO.Path]::GetFullPath($InputReceipt))) 'decoder-cql-metrics.jsonl'
    $taskRows=@(Get-Content -LiteralPath $taskMetrics -Encoding UTF8|Where-Object {$_.Trim().Length}|ForEach-Object {ConvertFrom-Json -InputObject $_})
    if($taskRows.Count -lt 1){throw 'No sampled CQL training metrics to plot'}
    $taskCqlOutput=[IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
    if(-not [IO.Directory]::Exists([IO.Path]::GetDirectoryName($taskCqlOutput))){throw 'Plot output parent must exist'}
    $taskFigure=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    try{
        $taskFigure.Width=1200;$taskFigure.Height=760;$taskFigure.BackColor=[Drawing.Color]::White
        $taskFigure.AntiAliasing='All';$taskFigure.TextAntiAliasingQuality='High'
        $taskFigure.Font=New-Object Drawing.Font('Microsoft YaHei',10)
        $taskMax=($taskRows|Measure-Object cursor -Maximum).Maximum/1000000.0
        $taskStep=if($taskMax -ge 1){0.25}else{0.1}
        foreach($taskAreaName in @('values','loss')){
            $taskArea=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea($taskAreaName)
            $taskArea.AxisX.Minimum=0;$taskArea.AxisX.Maximum=[Math]::Ceiling($taskMax/$taskStep)*$taskStep;$taskArea.AxisX.Interval=$taskStep
            $taskArea.AxisX.LabelStyle.Format='0.00';$taskArea.AxisX.Title='已训练采样（百万条）'
            $taskArea.AxisY.Title=if($taskAreaName -eq 'values'){'Q 值与最高终局回报'}else{'TD 与 CQL 损失分量'}
            $taskArea.AxisY.LabelStyle.Format='0.00'
            $taskArea.AxisX.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#e5e7eb')
            $taskArea.AxisY.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#e5e7eb')
            [void]$taskFigure.ChartAreas.Add($taskArea)
        }
        $taskFigure.ChartAreas['values'].Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(3,17,46,65)
        $taskFigure.ChartAreas['loss'].Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(51,17,46,65)
        $taskDefinitions=@(
            @{name='日志批次 Q 均值';field='Q';area='values';color='#2563eb'},
            @{name='监控批次 Q 的 95 分位';field='Q/P95';area='values';color='#d97706'},
            @{name='Huber TD 损失';field='Loss (TD Error)';area='loss';color='#2563eb'},
            @{name='加权 CQL 正则项';field='Loss (Regularizer)';area='loss';color='#059669'}
        )
        foreach($taskDefinition in $taskDefinitions){
            $taskSeries=New-Object System.Windows.Forms.DataVisualization.Charting.Series($taskDefinition.name)
            $taskSeries.ChartArea=$taskDefinition.area;$taskSeries.ChartType='Line';$taskSeries.BorderWidth=2
            $taskSeries.Color=[Drawing.ColorTranslator]::FromHtml($taskDefinition.color)
            foreach($taskRow in $taskRows){
                $taskY=$taskRow.metrics.($taskDefinition.field)
                if($null -ne $taskY){[void]$taskSeries.Points.AddXY($taskRow.cursor/1000000.0,[double]$taskY)}
            }
            [void]$taskFigure.Series.Add($taskSeries)
        }
        $taskBound=New-Object System.Windows.Forms.DataVisualization.Charting.Series('最高终局回报 1.5')
        $taskBound.ChartArea='values';$taskBound.ChartType='Line';$taskBound.BorderWidth=2;$taskBound.BorderDashStyle='Dash'
        $taskBound.Color=[Drawing.ColorTranslator]::FromHtml('#dc2626')
        [void]$taskBound.Points.AddXY(0,1.5);[void]$taskBound.Points.AddXY($taskMax,1.5)
        [void]$taskFigure.Series.Add($taskBound)
        $taskTitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
        $taskTitle.Text='稠密解码器 CQL · 训练诊断';$taskTitle.Font=New-Object Drawing.Font('Microsoft YaHei',19,[Drawing.FontStyle]::Bold)
        [void]$taskFigure.Titles.Add($taskTitle)
        $taskSubtitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
        $taskSubtitle.Text=("BC 骨干，独立 Q 头 | alpha={0}，gamma=1，整场最终顺位奖励" -f $taskPlan.alpha)
        $taskSubtitle.Font=New-Object Drawing.Font('Microsoft YaHei',11)
        [void]$taskFigure.Titles.Add($taskSubtitle)
        $taskLegend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend
        $taskLegend.Docking='Bottom';$taskLegend.Font=New-Object Drawing.Font('Microsoft YaHei',9)
        [void]$taskFigure.Legends.Add($taskLegend)
        $taskFooter=New-Object System.Windows.Forms.DataVisualization.Charting.Title
        $taskFooter.Docking='Bottom';$taskFooter.Font=New-Object Drawing.Font('Microsoft YaHei',9)
        $taskFooter.Text='曲线为日志批次采样，非全体样本统计；小 TD 损失不能排除价值过估计，强度以独立配对对局为准。'
        [void]$taskFigure.Titles.Add($taskFooter)
        $taskFigure.SaveImage($taskCqlOutput,'Png')
        Write-Output $taskCqlOutput
    }finally{$taskFigure.Dispose()}
    return
}
$taskPoints=@($taskReceipt.validation_curve)
if($taskPoints.Count -lt 1){throw 'No complete validation measurements to plot'}
$taskOutput=[IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
if(-not [IO.Directory]::Exists([IO.Path]::GetDirectoryName($taskOutput))){throw 'Plot output parent must exist'}
$taskChart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
try {
    $taskChart.Width=1200;$taskChart.Height=760
    $taskChart.BackColor=[Drawing.Color]::White
    $taskChart.AntiAliasing='All';$taskChart.TextAntiAliasingQuality='High'
    $taskFont=New-Object Drawing.Font('Microsoft YaHei',10)
    $taskChart.Font=$taskFont
    $taskMaximum=[Math]::Max(0.01,($taskPoints|Measure-Object -Property cursor -Maximum).Maximum/1000000.0)
    foreach($taskName in @('nll','accuracy')){
        $taskArea=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea($taskName)
        $taskArea.BackColor=[Drawing.Color]::White
        $taskInterval=if($taskMaximum -ge 1){0.5}else{0.1}
        $taskArea.AxisX.Minimum=0;$taskArea.AxisX.Maximum=[Math]::Ceiling($taskMaximum/$taskInterval)*$taskInterval
        $taskArea.AxisX.Interval=$taskInterval
        $taskArea.AxisX.Title='已训练决策数（百万条）';$taskArea.AxisX.LabelStyle.Format='0.0'
        $taskArea.AxisY.Minimum=0
        $taskArea.AxisX.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#e5e7eb')
        $taskArea.AxisY.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#e5e7eb')
        $taskArea.AxisX.LineColor=[Drawing.Color]::Silver;$taskArea.AxisY.LineColor=[Drawing.Color]::Silver
        if($taskName -eq 'nll'){
            $taskArea.AxisY.Title='验证 NLL（越低越好）'
            $taskArea.AxisY.Maximum=[Math]::Ceiling(($taskPoints|Measure-Object negative_log_likelihood -Maximum).Maximum*2)/2
            $taskArea.AxisY.Interval=0.5
            $taskArea.AxisY.LabelStyle.Format='0.00'
        }else{
            $taskArea.AxisY.Title='验证动作一致率（%）';$taskArea.AxisY.Maximum=100
        }
        [void]$taskChart.ChartAreas.Add($taskArea)
    }
    $taskChart.ChartAreas['nll'].Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(3,17,46,65)
    $taskChart.ChartAreas['accuracy'].Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(51,17,46,65)
    $taskTitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $taskTitle.Text='同规模 BC · 验证集学习曲线'
    $taskTitle.Font=New-Object Drawing.Font('Microsoft YaHei',19,[Drawing.FontStyle]::Bold)
    [void]$taskChart.Titles.Add($taskTitle)
    $taskSubtitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $taskSubtitle.Text=('稠密解码器 {0:N0} 参数 | 相同 5,120 局与原 train/validation/test 划分' -f $taskPlan.parameters)
    $taskSubtitle.Font=New-Object Drawing.Font('Microsoft YaHei',11)
    [void]$taskChart.Titles.Add($taskSubtitle)
    $taskLegend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend
    $taskLegend.Docking='Bottom';$taskLegend.Font=$taskFont
    [void]$taskChart.Legends.Add($taskLegend)
    foreach($taskName in @('nll','accuracy')){
        $taskSeries=New-Object System.Windows.Forms.DataVisualization.Charting.Series("decoder-$taskName")
        $taskSeries.ChartArea=$taskName;$taskSeries.ChartType='Line';$taskSeries.BorderWidth=3
        $taskSeries.Color=[Drawing.ColorTranslator]::FromHtml('#2563eb')
        $taskSeries.MarkerStyle='Circle';$taskSeries.MarkerSize=7
        $taskSeries.LegendText='当前 decoder BC'
        if($taskName -eq 'accuracy'){$taskSeries.IsVisibleInLegend=$false}
        foreach($taskPoint in $taskPoints){
            $taskY=if($taskName -eq 'nll'){$taskPoint.negative_log_likelihood}else{100*$taskPoint.action_accuracy}
            [void]$taskSeries.Points.AddXY($taskPoint.cursor/1000000.0,$taskY)
        }
        [void]$taskChart.Series.Add($taskSeries)
        $taskReference=New-Object System.Windows.Forms.DataVisualization.Charting.Series("old-$taskName")
        $taskReference.ChartArea=$taskName;$taskReference.ChartType='Line';$taskReference.BorderWidth=2;$taskReference.BorderDashStyle='Dash'
        $taskReference.Color=[Drawing.ColorTranslator]::FromHtml('#64748b')
        $taskReference.LegendText='旧 encoder BC（完整训练后的参考值）'
        if($taskName -eq 'accuracy'){$taskReference.IsVisibleInLegend=$false}
        $taskY=if($taskName -eq 'nll'){$taskPlan.old_bc_metrics.validation.negative_log_likelihood}else{100*$taskPlan.old_bc_metrics.validation.action_accuracy}
        [void]$taskReference.Points.AddXY(0,$taskY);[void]$taskReference.Points.AddXY($taskMaximum,$taskY)
        [void]$taskChart.Series.Add($taskReference)
    }
    $taskFooter=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $taskFooter.Docking='Bottom';$taskFooter.Font=New-Object Drawing.Font('Microsoft YaHei',9)
    $taskFooter.Text='每个点均读取完整验证集（152,703 条决策）；结构和优化器方案同时变化，不能归因于单一改动。'
    [void]$taskChart.Titles.Add($taskFooter)
    $taskChart.SaveImage($taskOutput,'Png')
    Write-Output $taskOutput
} finally {$taskChart.Dispose()}
