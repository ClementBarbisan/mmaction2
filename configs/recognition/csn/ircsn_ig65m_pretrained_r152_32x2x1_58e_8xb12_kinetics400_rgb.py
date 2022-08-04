_base_ = [
    'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb.py'
]

model = dict(
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'))
