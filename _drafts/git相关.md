---
title: git相关
layout: posts
---



在终端依次输入下面两句命令：

```
git config --global credential.helper store

git pull /git push (第一次输入，后续就不用再次数据)

push/pull 你的代码 , 这时会让你输入用户名和密码, 这一步输入的用户名密码会被记住, 下次再push/pull 代码时就不用输入用户名和密码了。
```

 这一步是在用户目录下生成文件.git-credential 记录用户名和密码信息