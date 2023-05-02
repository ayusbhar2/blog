cd ~/Desktop/Career/blog_posts/blog

# convert the overleaf project's main.tex file to main.md file using pandoc
pandoc -o ../neural_network_verification_using_linear_programming/main.md ../neural_network_verification_using_linear_programming/main.tex

cp ../neural_network_verification_using_linear_programming/main.md ./_posts/neural_network_verification_using_linear_programming.md

mkdir ./images/neural_network_verification_using_linear_programming

cp -R ../neural_network_verification_using_linear_programming/images ./images/neural_network_verification_using_linear_programming/images

