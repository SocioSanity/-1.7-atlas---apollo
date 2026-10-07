$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class AtlasUiNative {
  [DllImport("user32.dll", CharSet=CharSet.Unicode)]
  public static extern IntPtr FindWindow(string cls, string title);
  [DllImport("user32.dll")]
  public static extern IntPtr GetDlgItem(IntPtr parent, int id);
  [DllImport("user32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern bool SetWindowText(IntPtr hwnd, string text);
  [DllImport("user32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern IntPtr SendMessageTimeout(IntPtr hwnd, uint msg, IntPtr wParam, string text, uint flags, uint timeout, out IntPtr result);
  [DllImport("user32.dll")]
  public static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
  [DllImport("user32.dll")]
  public static extern IntPtr SendMessage(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
  [DllImport("user32.dll")]
  public static extern bool IsWindowEnabled(IntPtr hwnd);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)]
  public static extern int GetWindowTextLength(IntPtr hwnd);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)]
  public static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int capacity);
}
'@

$running = Get-Process atlas_1_0_chat -ErrorAction SilentlyContinue |
  Where-Object { $_.MainWindowHandle -ne 0 } |
  Sort-Object StartTime -Descending | Select-Object -First 1
if (-not $running) { throw 'Atlas chat window is not open.' }
$window = [IntPtr]$running.MainWindowHandle
$prompt = [AtlasUiNative]::GetDlgItem($window, 101)
$send = [AtlasUiNative]::GetDlgItem($window, 102)
$output = [AtlasUiNative]::GetDlgItem($window, 103)
if ($prompt -eq [IntPtr]::Zero -or $send -eq [IntPtr]::Zero -or $output -eq [IntPtr]::Zero) {
  throw 'Atlas chat controls were not found.'
}
$existing = New-Object System.Text.StringBuilder 4096
[void][AtlasUiNative]::GetWindowText($prompt, $existing, $existing.Capacity)
$question = 'Reply with exactly one word: READY.'
if ($existing.Length -gt 0 -and $existing.ToString() -ne $question) {
  throw 'The chat input contains different text; leaving it untouched.'
}
$before = [AtlasUiNative]::GetWindowTextLength($output)
[IntPtr]$ignored = [IntPtr]::Zero
$setResult = [AtlasUiNative]::SendMessageTimeout($prompt, 0x000C, [IntPtr]::Zero, $question, 0x0002, 1000, [ref]$ignored)
if ($setResult -eq [IntPtr]::Zero) { throw 'Could not set the chat prompt through its window control.' }
[void][AtlasUiNative]::PostMessage($window, 0x0111, [IntPtr]102, $send)
Start-Sleep -Milliseconds 700
$deadline = (Get-Date).AddSeconds(90)
$wasDisabled = $false
while ((Get-Date) -lt $deadline) {
  $enabled = [AtlasUiNative]::IsWindowEnabled($send)
  if (-not $enabled) { $wasDisabled = $true }
  if ($wasDisabled -and $enabled) { break }
  Start-Sleep -Milliseconds 500
}
$length = [AtlasUiNative]::GetWindowTextLength($output)
$capacity = [Math]::Min([Math]::Max($length + 1, 1), 60000)
$text = New-Object System.Text.StringBuilder $capacity
[void][AtlasUiNative]::GetWindowText($output, $text, $capacity)
$full = $text.ToString()
$tailStart = [Math]::Min($before, $full.Length)
$tail = $full.Substring($tailStart)
if (-not $wasDisabled -or -not [AtlasUiNative]::IsWindowEnabled($send)) {
  throw 'Atlas did not finish within 90 seconds.'
}
Set-Content -LiteralPath 'atlas_ui_design_response_temp.txt' -Value $tail -Encoding utf8
Write-Output $tail
