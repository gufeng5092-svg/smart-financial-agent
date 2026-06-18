# GitHub 发布检查清单

发布前建议逐项确认：

- [ ] `.env` 没有被提交。
- [ ] 代码中没有真实 API Key、数据库密码和本机绝对路径。
- [ ] 完整比赛附件、研报 PDF、商业财报数据未被提交，或已确认允许公开。
- [ ] `README.md` 中说明了项目定位、安装方式、运行命令和数据准备方式。
- [ ] `requirements.txt` 能安装所有运行依赖。
- [ ] `sql/schema.sql` 能创建基础表结构。
- [ ] `examples/` 中仅包含可公开的样例输入输出。
- [ ] 至少跑过 `python -m py_compile *.py`。
- [ ] 初始化 Git 仓库后检查 `git status`，确认没有误提交生成物。
