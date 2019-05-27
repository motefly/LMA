for model in 0 1 2
do
    python main_opennmt.py -display_metrics -manager nips_0512 -stModel $model -stud_act aplu
    python main_opennmt.py -display_metrics -manager nips_0512 -stModel $model -stud_act relu
    python main_opennmt.py -display_metrics -manager nips_0512 -stModel $model -stud_act prelu
    python main_opennmt.py -display_metrics -manager nips_0512 -stModel $model -stud_act swish
    python main_opennmt.py -display_metrics -manager nips_0512 -stModel $model -stud_act lma
done
