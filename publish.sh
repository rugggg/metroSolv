#!/bin/bash

jupyter-nbconvert --to slides metro.ipynb --reveal-prefix=reveal.js
mv slides.slides.html  index.html
mkdir -p /tmp/workspace
cp -r * /tmp/workspace/
git add -A .
git commit -m "Update"
git checkout -B gh-pages
cp -r /tmp/workspace/* .
git add -A .
git commit -m "new version"
git push -f origin master gh-pages
git checkout master
rm -rf /tmp/workspace
