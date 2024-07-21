# PowerShell脚本内容

$Command = "C:/Users/qin/anaconda3/envs/py3/python.exe "  # 要执行的命令或程序路径
$Arguments = "E:\jikewct\Repos\ml_code/main.py " # 命令的参数（如果有的话）
#$Arguments = "E:\jikewct\Repos\ml_code/test.py " # 命令的参数（如果有的话）

# 使用Invoke-Expression在后台运行命令
#Invoke-Expression "& '$Command' $Arguments > output.txt 2> error.txt"

# 脚本结束，但命令继续在后台运行
echo $args
#Set-Location -Path  "E:/jikewct/Repos/ml_code/" -PassThru
#Invoke-Expression " & 'Invoke-WmiMethod' -Path 'Win32_Process' -Name  Create -ArgumentList '$Command $Arguments $args' "
Invoke-Expression " & 'Invoke-WmiMethod' -Path 'Win32_Process' -Name  Create  -ArgumentList '$Command $Arguments $args' "
