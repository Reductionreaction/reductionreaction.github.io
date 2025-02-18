cd _posts
tempname=$(date "+%Y-%m-%d")
filename=$tempname-$1.md
touch $filename
cat > $filename <<EOF
---
layout: post
title: $1
date: $tempname
---
EOF

