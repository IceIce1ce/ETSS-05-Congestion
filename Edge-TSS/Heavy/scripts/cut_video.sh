mkdir -p output_30s
for file in *.mp4; do
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
  if (( $(echo "$duration > 30" | bc -l) )); then
    echo "Cutting $file (${duration}s)"
    ffmpeg -i "$file" -t 30 -c copy -y "output_30s/$file"
  else
    echo "Skipping $file (${duration}s - already short enough)"
  fi
done