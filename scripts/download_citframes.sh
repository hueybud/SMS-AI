#!/bin/bash
# Downloads all .citframes files from each player's Google Drive folder
# into a single destination directory.
# First time: chmod +x download_citframes.sh
# Usage: ./download_citframes.sh /path/to/destination (ex. ./download_citframes.sh ~/citframes)

DEST=${1:-./citframes}
mkdir -p "$DEST"

declare -A FOLDERS=(
    ["Pied"]="1CEv0lvxnaeFoMZQk5YkmSpN7aSgFYNBG"
    ["TheSweetieMan"]="1YtwVMzwVc3R4I3CIvkLoTO8zZz2JSNEH"
    ["Ein"]="1b9cM3r3DET4ujYJ75neft18FJzwnHCCJ"
    ["DanK"]="1zo6V1Esd-GWSUmFyQ6vuduz_p_5g2FNm"
    ["Febreze"]="13XPHDequ7dbhHKwwIj7vQwxa42ebT34G"
    ["XanderG"]="1SH3tIgEq322iUj5IilrBZuhK8JinnAZD"
)

for PLAYER in "${!FOLDERS[@]}"; do
    ID="${FOLDERS[$PLAYER]}"
    echo "=== Downloading $PLAYER ($ID) ==="
    rclone copy gdrive: "$DEST" \
        --drive-root-folder-id="$ID" \
        --include "*.citframes" \
        --ignore-existing \
        --progress \
        --transfers 8
done

echo "=== Done. Files in $DEST ==="
ls "$DEST" | wc -l
