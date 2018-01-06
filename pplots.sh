python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results01/nn03/active --plot -d ab_active.pdf --legend-columns 2
python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results01/nn03/ignore --plot -d ab_ignore.pdf --legend-columns 2
python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results01/nn03/none --plot -d ab_absorb.pdf --legend-columns 2

python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results02/nn03/ignore --plot -d e_ignore.pdf
python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results02/nn03/none --plot -d e_absorb.pdf
python config_selection/display_config_scores_mnist.py -b ../mnist_models_lrp/pertubation/results02/nn03/active --plot -d e_active.pdf

