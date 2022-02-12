---
layout: post
title: A short beginning
date: 2021-12-23 15:59:00-0400
inline: false
---

A short beginning.

***

# 前言

第一篇blog献给踏入搭建博客的缺德河流。

首先要说明本篇文章适合人群：对Github了解不甚深的人。如果你掌握Git，知道仓库是什么，会写网站，以下文章或许更适合你：


- [使用 github + jekyll 搭建个人博客](https://www.cnblogs.com/wangfupeng1988/p/5702324.html)
- [Github Pages + Jekyll 搭建个人博客主页](https://leyuanheart.github.io/2020/06/06/Github-Pages+Jekyll/)
- [Github+Jekyll搭建个人博客](https://blog.csdn.net/White_Idiot/article/details/69397224/)


在写blog的时候试图叙述清楚整个操作流程，如果有幸运观众看到了这里，就请你跟我做起来。21世纪人人都应该有blog，没有blog的人滚出🌏！(夸张手法)


# 背景知识

## Github，Github Pages，Git，Repository，Fork，jekyll，Markdown.

### Github
谷歌学术的引言是：“Stand on the shoulders of giants”。Github就是给巨人加了个垫肩，让人站的比较稳。说到这里，穿插一个小故事，因为和故事中另一个主角已经绝交，担心以后忘记。我在申请期间写文书的时候发现很多同学不愿意分享自己的文书，郁闷之际和朋友说起这件事并发表：**获得知识的途径不应该有壁垒**，并说出：我要创建一个共享文书的网站。朋友对此持反对意见并表示：我也不会分享自己的文书。当知道他的文书价值8万的时候我表示理解，并更改了自己的观点：**获得知识的途径不应该有很强壁垒**。或许当大多数人能分辨出“我在普林斯顿上课”和“我上过普林斯顿的公开课”的区别，强字就可以去掉了。

### Github Pages
Github Pages是一个静态网站托管服务，我理解的是可以通过这个服务，提交代码生成网站。

### Git
官方：Git是一个免费、开源的分布式版本控制系统。

我眼里的Git:一些操作命令。

### Repository
中文翻译为仓库，可以理解为项目，类比为文件夹。

### Fork
学习是站在巨人的肩膀上，创新就是加入自己的修改。Fork他人的项目就是基于他人的代码加入自己的coding。(本土🐕今天才知道是什么)

### jekyll
类比word template,不用学习HTML，只需会打字，Jekyll负责转换。

### Markdown
官方：排版工具

我眼里的Markdown: 逊于LaTeX的排版工具。



# 流程

创造的关键事件有：创建Github项目；安装jekyll；个性化博客；提交到Github。

更新博客的流程可以归纳为：本地写Markdown，然后push到Github。


**① 创建github项目**

全程copy官网[Github Pages](https://pages.github.com/)。除了官网，还要感谢王老板的帮助。

新建一个仓库，命名为`github用户名.github.io`

![image-1](https://pic.downk.cc/item/5f095e6514195aa5942e34a3.png)


在仓库里新建一个`index.html`文件，输入上图的代码。
![image-2](https://pic.downk.cc/item/5f095e6514195aa5942e34a7.png)


在浏览器里输入`username.github.io`，就能看到`index.html`文件解析的结果。解析成功这一part就完成了。

![image-3](https://pic.downk.cc/item/5f095e6514195aa5942e34ad.png)
(此段参考的最上🔗2)


**② 安装 jekyll**

Ruby是Jekyll的编程语言，所以安装Jekyll之前需要先安装Ruby和相应的DevKit。

不管三七二十一，打开[ruby installer](https://rubyinstaller.org/downloads/)，选择右边推荐的下载。

![image-4](https://pic.imgdb.cn/item/61c44cff2ab3f51d919c7955.png)


下载完成后运行

```bash
gem install jekyll
```
查看版本号：
```bash
jekyll -v
```
如果可以看到版本号说明安装成功噜。

接下来就在本地测试一下：

```bash
# 安装bundler，bundler通过Gemfile文件来管理gem包
gem install  bundler
# 创建一个新的Jekyll项目，并命名为myblog
jekyll new myblog
# 进入myblog目录
cd myblog
# 创建本地服务器，默认的运行地址为http://localhost:4000
# bundle exec 表示在当前项目依赖的上下文环境中执行命令 jekyll serve
bundle exec jekyll serve
```

在浏览器里输入地址 http://localhost:4000，可以看到：
![image-8](https://pic.downk.cc/item/5f095f2414195aa5942e6d28.png)


恭喜你，成功一大半了！


(此段图片参考的最上🔗2)


**③ 个性化博客**

此时就可以随意发挥了。可以在jekyll官网或者Github搜索模板。我使用了[huxblog-boilerplate](https://github.com/Huxpro/huxblog-boilerplate)的模板，如下

![image-8](https://pic.imgdb.cn/item/61c46ab62ab3f51d91a781b1.png)

可以fork也可以下载到本地更改_config等文件。此处非常感谢网友[Ye yuan](https://leyuanheart.github.io/)。(虽然网友并不知情就是了，但这就是开源世界的美好！)

**④ 提交到github**

把about,_config文件修改完成之后保存。

cd到需要提交的目录下，执行：

```bash
$ git add .
$ git commit -m "statement"   //此处statement填写此次提交修改的内容，作为日后查阅
$ git push origin master
```

前方一路顺畅，在这里遇到了一个大坑^^

在提交的时候密码怎么都不对，重置了密码也不对，所以开始怀疑以上几步是否正确。经过一天的挣扎，发现原来是：[密码换成了token](http://codetd.com/article/13580067)。

同步仓库(好习惯是一生的本钱)：

```bash
$ git clone https://github.com/{username}/{username}.github.io.git  
```

浏览器输入自己的Github Pages地址，如果和本地预览的一样，太恭喜你啦，成功啦！



### 遇到的问题(持续更新，问题总是无穷的)
* 上传表格的时候显出不出来，参考这位网友：
[MarkDown中的表格在jekyll的pages博客中不能正常显示](https://blog.csdn.net/sdujava2011/article/details/83692576)

* 在Github页面版修改了readme文件再本地上传时候会提示错误，强制push即可。

* 图片上传使用新浪微博里的链接解析不了，还是老老实实用图床把。

* 公式编不出来，可以参考：[如何在基于jekyll的github上发布的博客中支持MathJax(LaTex数学公式)](https://www.zhihu.com/question/62114522?sort=created)


### To-do List

- 目录

- 分享链接🔗

- 学会Fork






