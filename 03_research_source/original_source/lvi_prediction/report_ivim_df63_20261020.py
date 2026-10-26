# -*- coding: utf-8 -*-
"""All63 results, pretest nested choice, andexplicit test-screening limitations."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve
import ivim_df63_study_20261020 as study
import run_ivim_df63_20261020 as run
from report_regional_summary77_two_tests_20261014 import table

OUT = run.OUT
REPORT = OUT/'IVIM_D和f六维63组合_两组100例对照报告_20261020.md'
NORMS = {'PercentileZ': '截尾＋Z-score', 'RobustZ': '稳健Z-score'}
SHORT = {'D_Intensity': 'D-I', 'D_Mean3mm': 'D-M', 'D_SD3mm': 'D-S',
         'f_Intensity': 'f-I', 'f_Mean3mm': 'f-M', 'f_SD3mm': 'f-S'}


def description(combo):
    row = next(r for r in study.feature.combinations63() if r['ComboID'] == combo)
    return '+'.join(SHORT[d] for d in row['Dimensions'])


def metrics_ci(row):
    return f'{row.AUC:.4f}（{row.AUC_CI95Low:.4f}—{row.AUC_CI95High:.4f}）'


def merge():
    choices = study.read(OUT, 'fixed_training_choices')
    metrics = study.read(OUT, 'test_metrics')
    metrics = metrics[~metrics.ModelKey.str.startswith('Adaptive')].copy()
    metrics['Normalization'] = metrics.ModelKey.str.split('__').str[0]
    metrics['ComboID'] = metrics.ModelKey.str.split('__').str[1]
    merged = choices.merge(metrics.drop(columns='Threshold'), on=['Group', 'Normalization', 'ComboID'], validate='one_to_one')
    assert len(merged) == 252
    merged.to_csv(OUT/'all63_two_samples_detailed_comparison.csv', index=False, encoding='utf-8-sig')
    summary = []
    indexed = merged.set_index(['Group', 'Normalization', 'ComboID'])
    for norm in study.NORMS:
        for c in study.feature.combinations63():
            a, b = [indexed.loc[(group, norm, c['ComboID'])] for group in study.GROUPS]
            summary.append({'Normalization': norm, 'ComboID': c['ComboID'], 'DimensionCount': c['DimensionCount'],
                'Dimensions': '+'.join(c['Dimensions']), 'ShortName': description(c['ComboID']),
                **{group+'_'+field: float(r[field]) for group, r in [('SampleA', a), ('SampleB', b)]
                    for field in ('TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC', 'AUC', 'AUC_CI95Low', 'AUC_CI95High', 'FeatureCount')},
                'MeanOuterAUC': float((a.OuterMeanAUC+b.OuterMeanAUC)/2),
                'MinimumOuterAUC': float(min(a.OuterMeanAUC, b.OuterMeanAUC)),
                'MeanTestAUC': float((a.AUC+b.AUC)/2), 'AbsoluteTestDifference': float(abs(a.AUC-b.AUC))})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'all63_side_by_side_summary.csv', index=False, encoding='utf-8-sig')
    return merged, summary


def training_support(summary, merged, adaptive):
    # Descriptive cross-sample shortlist only; neveruses testprobabilities orAUCforranking.
    inner = study.read(OUT, 'adaptive_inner_selections')
    joint = inner[inner.ModelKey == 'Adaptive__BothNorms']
    records = []
    for _, row in summary.iterrows():
        matching = joint[(joint.normalization == row.Normalization)&(joint.ComboID == row.ComboID)]
        fixed = merged[(merged.Normalization == row.Normalization)&(merged.ComboID == row.ComboID)]
        records.append({**row.to_dict(), 'FullTrainingSelectedCount': int((matching.Phase == 'final').sum()),
            'OuterInnerSelectedCount': int((matching.Phase != 'final').sum()),
            'AvailableInBothFinalTrainings': bool(fixed.AvailableFinal.all() and fixed.AvailableAllInnerFolds.all())})
    ranked = pd.DataFrame(records).sort_values(['AvailableInBothFinalTrainings', 'FullTrainingSelectedCount',
        'OuterInnerSelectedCount', 'MinimumOuterAUC', 'MeanOuterAUC', 'DimensionCount', 'Normalization', 'ComboID'],
        ascending=[False, False, False, False, False, True, True, True])
    ranked.to_csv(OUT/'training_only_followup_shortlist.csv', index=False, encoding='utf-8-sig')
    selected = ranked[ranked.AvailableInBothFinalTrainings].head(3)
    study.save(OUT/'followup_candidates.json', {'purpose': 'prespecified training-only descriptive priority forfutureconfirmation',
        'rank_fields': ['bothfinalavailable', 'fulltrainingjointselection count', 'outerinnerjointselection count',
            'minimumouterAUC', 'meanouterAUC', 'lowerdimensioncount', 'normalization', 'comboID'],
        'test_AUC_used_for_ranking': False, 'cross_sample_selection_not_independently_validated': True,
        'formal_model_adoption': False, 'candidates': selected[['Normalization', 'ComboID', 'Dimensions',
            'FullTrainingSelectedCount', 'OuterInnerSelectedCount', 'MinimumOuterAUC', 'MeanOuterAUC']].to_dict('records')})
    return selected


def figures(summary, adaptive, metrics, predictions):
    font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 10, 'axes.unicode_minus': False})
    directory = OUT/'figures'
    directory.mkdir(exist_ok=True)
    files = []
    for norm in study.NORMS:
        rows = summary[summary.Normalization == norm].set_index('ComboID')
        for part in range(3):
            names = [c['ComboID'] for c in study.feature.combinations63()[part*21:(part+1)*21]]
            current = rows.loc[names]
            cols = [g+'_'+field for g in study.GROUPS for field in ('TrainFitAUC', 'OuterMeanAUC', 'AUC')]
            values = current[cols].to_numpy(float)
            fig, ax = plt.subplots(figsize=(12.8, 9.3))
            fig.subplots_adjust(left=.24, right=.88, top=.95, bottom=.07)
            im = ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
            ax.set(xticks=np.arange(6), xticklabels=['A 训练拟合', 'A 外层均值', 'A 测试', 'B 训练拟合', 'B 外层均值', 'B 测试'],
                yticks=np.arange(21), yticklabels=[name+'  '+description(name) for name in names],
                title=NORMS[norm]+'：两组100例，'+names[0]+'—'+names[-1])
            ax.tick_params(axis='y', labelsize=8)
            for i in range(21):
                for j in range(6):
                    ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center',
                        color='white' if values[i, j] > .67 else '#13252a', fontsize=9)
            ax.axvline(2.5, color='white', lw=2)
            color_axis = fig.add_axes([.92, .22, .016, .58])
            fig.colorbar(im, cax=color_axis, label='AUC')
            name = norm+'_63组合_第'+str(part+1)+'部分'
            for ext in ('png', 'svg'):
                fig.savefig(directory/(name+'.'+ext), dpi=170)
            plt.close(fig);files.append(name+'.png')
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    for ax, group in zip(axes, study.GROUPS):
        for key, color, ls in [('Adaptive__PercentileZ', '#aa742f', '-'), ('Adaptive__RobustZ', '#237d78', '-'), ('Adaptive__BothNorms', '#555555', '--')]:
            p = predictions[(predictions.Group == group)&(predictions.ModelKey == key)]
            fpr, tpr, _ = roc_curve(p.Outcome, p.Probability)
            c = adaptive.loc[(group, key)]
            name = '联合选择' if key.endswith('BothNorms') else NORMS[c.Normalization]
            auc = metrics.loc[(group, key), 'AUC']
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.7, label=name+' '+c.ComboID+f' AUC={auc:.3f}')
        ax.plot([0, 1], [0, 1], ':', color='#999999')
        ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel='1-特异度', ylabel='敏感度', title=group+'：训练内选组合后冻结测试')
        ax.legend(loc='lower right', fontsize=9);ax.grid(alpha=.15)
    name = '训练内选择63组合_冻结测试ROC'
    for ext in ('png', 'svg'):
        fig.savefig(directory/(name+'.'+ext), dpi=170)
    plt.close(fig);files.append(name+'.png')
    study.save(OUT/'figure_manifest.json', {'png_files': files, 'model_predict_calls': 0})
    return files


def audit_extraction():
    parts, errors, normalization = [], [], []
    for index in study.union_indices('sample'):
        row = run.minroi.canonical(joblib.load(study.patient_path(index)))
        parts.extend(row['partition_audit'])
        errors.extend(row['texture_errors'])
        normalization.extend({'AnalysisID': row['AnalysisID'], **r} for r in row['normalization_audit'])
    run.csv('partition_and_feature_audit', parts)
    pd.DataFrame(errors, columns=['AnalysisID', 'Normalization', 'ComboID', 'Variant', 'Channel', 'Habitat', 'Error']).to_csv(OUT/'texture_errors.csv', index=False, encoding='utf-8-sig')
    run.csv('normalization_audit', normalization)
    df = pd.DataFrame(parts)
    summary = df.groupby(['Normalization', 'ComboID', 'Variant']).agg(PatientN=('AnalysisID', 'size'),
        MeanSmallHabitatCount=('SmallHabitatCount', 'mean'), MinimumFiniteFeatures=('FiniteFeatureCount', 'min')).reset_index()
    run.csv('feature_quality_summary', summary)
    return len(errors)


def main():
    run.frozen()
    assert json.loads((OUT/'inference_verification.json').read_text(encoding='utf-8'))['passed']
    merged, summary = merge()
    adaptive = study.read(OUT, 'adaptive_training_choices').set_index(['Group', 'ModelKey'])
    metrics = study.read(OUT, 'test_metrics').set_index(['Group', 'ModelKey'])
    pairs = study.read(OUT, 'paired_combo_vs_all6_auc').set_index(['Group', 'Normalization', 'ComboID'])
    shortlist = training_support(summary, merged, adaptive)
    balanced = summary.assign(MinimumTestAUC=summary[['SampleA_AUC', 'SampleB_AUC']].min(axis=1)).sort_values(
        ['MinimumTestAUC', 'MeanOuterAUC', 'DimensionCount', 'Normalization', 'ComboID'],
        ascending=[False, False, True, True, True])
    balanced.to_csv(OUT/'posthoc_balanced_test_screening.csv', index=False, encoding='utf-8-sig')
    study.save(OUT/'posthoc_balanced_screening_disclosure.json', {'test_AUC_used_for_ranking': True,
        'rule': 'posthoc minimumof two testAUC descending,meanouter descending,dimensioncountascending',
        'scope': 'exploratoryfuturecandidate only;not pretestselected orindependently validated',
        'formal_adoption': False})
    files = figures(summary, adaptive, metrics, study.read(OUT, 'test_predictions'))
    error_n = audit_extraction()
    overlap = json.loads((OUT/'between_sample_overlap.json').read_text(encoding='utf-8'))['counts']
    body = ['# IVIM D、f六维输入63组合：两组随机100例实验', '',
        '日期：2026年10月20日。用户确认全部63种非空组合、每组70训练／30测试、k＝3、仅IVIM；组合只改变聚类输入，聚类后统一测量D和f。当前为探索研究，不改变此前模型采用决定。', '',
        '## 本轮判断', '',
        f'全部126个标准化／组合方案中，没有方案在两组测试中均达到0.75。A组测试AUC范围{summary.SampleA_AUC.min():.3f}—{summary.SampleA_AUC.max():.3f}，B组为{summary.SampleB_AUC.min():.3f}—{summary.SampleB_AUC.max():.3f}；B组所有组合最高只有{summary.SampleB_AUC.max():.3f}。当前不宜确定可正式采用的最终组合。', '',
        '如只为下一轮固定探索候选，可同时保留下文训练内选中的方案，以及测试后表现较均衡的方案；二者选择依据不同，均需要后续确认。不能将一次测试筛查结果解释为已验证的最优性能。', '',
        '## 测试前由训练内选择锁定的方案', '',
        table(['数据组', '选择范围', '最终组合', '输入维度', '训练拟合AUC', '完整流程外层均值±SD', '汇总OOF AUC', '冻结30例测试AUC（95%区间）'],
            [[group, '联合选择→'+NORMS[c.Normalization] if key.endswith('BothNorms') else NORMS[c.Normalization], c.ComboID,
                description(c.ComboID), f'{c.TrainFitAUC:.4f}', f'{c.OuterMeanAUC:.4f}±{c.OuterSD:.4f}', f'{c.PooledOOFAUC:.4f}',
                metrics_ci(metrics.loc[(group, key)])] for (group, key), c in adaptive.iterrows()]), '',
        '本表的组合是在每组训练70例的内层五折中选择，并在测试前冻结。外层每折也只用当折的内层结果从63种组合中选优，因此外层指标评估完整的组合选择流程。它不等于事后挑出固定组合最高外层均值，也不等于63个测试AUC中的最大值。最终模型与对应固定组合共享测试概率，未新增预测调用。', '',
        table(['数据组', '选择范围', '最终标准化', '组合', '特征排序', '实际特征数', 'C', '冻结阈值'],
            [[group, key.replace('Adaptive__', ''), NORMS[c.Normalization], c.ComboID, c.ranking,
                int(c.FeatureCount), c.C, f'{c.Threshold:.6f}'] for (group, key), c in adaptive.iterrows()]), '',
        '## 下一轮优先复核候选', '',
        '以下只按训练记录确定优先级：两组最终训练联合选择次数、十个外层内层联合选择次数、较低组外层AUC、两组外层AUC均值，再优先较少维数；要求两组最终模型及所选配置内层均可用，不按测试AUC排序。这是供下一轮固定方案的探索性候选，不自动更改正式模型。', '',
        table(['优先级', '标准化', '组合', '输入', '最终训练选中次数／2', '外层内层选中次数／10', 'A／B外层均值', 'A／B测试'],
            [[i+1, NORMS[r.Normalization], r.ComboID, r.ShortName, int(r.FullTrainingSelectedCount), int(r.OuterInnerSelectedCount),
                f'{r.SampleA_OuterMeanAUC:.4f}／{r.SampleB_OuterMeanAUC:.4f}', f'{r.SampleA_AUC:.4f}／{r.SampleB_AUC:.4f}']
             for i, (_, r) in enumerate(shortlist.iterrows())]), '',
        '这个跨组汇总排序虽不直接使用测试AUC，但一组的部分测试病例在另一组属于训练病例，不能将上述候选本轮A/B测试表现解释为对该跨组选择的独立验证。各组独立、测试前锁定的选择结果仍以上表为准。', '',
        '## 测试后均衡性筛查：下一轮探索候选', '',
        '以下是看到全部测试结果后的补充探索：优先使两组中较低的测试AUC较高，再参考外层均值和较少维数。此排序直接使用了测试AUC，不属于预先锁定方案，也没有得到新的独立验证。它用于帮助决定下一轮比较哪些输入，不作为本轮正式模型采用依据。', '',
        table(['标准化', '组合', '输入', 'A／B拟合', 'A／B外层均值', 'A／B测试'],
            [[NORMS[r.Normalization], r.ComboID, r.ShortName,
                f'{r.SampleA_TrainFitAUC:.4f}／{r.SampleB_TrainFitAUC:.4f}',
                f'{r.SampleA_OuterMeanAUC:.4f}／{r.SampleB_OuterMeanAUC:.4f}',
                f'{r.SampleA_AUC:.4f}／{r.SampleB_AUC:.4f}'] for _, r in balanced.head(5).iterrows()]), '',
        '本轮最均衡的测试筛查方案为稳健Z-score＋C62：D局部均值、D局部SD、f强度、f局部均值、f局部SD。其两组拟合约0.697，外层均值0.633／0.589，测试0.675／0.670；测试95%区间约0.450—0.890／0.450—0.865。两个点估计接近尚不能证明稳定泛化，也未达到0.75目标。可以将C62、训练内候选C08及全六维C63保留为后续对照，当前不替换正式模型。', '',
        '## 63种组合的定义与模型含义', '',
        'D-I＝D强度，D-M＝D的3 mm局部均值，D-S＝D的3 mm局部SD；f-I、f-M、f-S为f对应三项。非空组合总数2⁶−1＝63，按维数分别为6、15、20、15、6、1种。C63包含全部六维。完整索引见combinations63.csv。', '',
        '本实验单独检验用于分区的输入维度。“仅D强度”仍在生成的生境内提取D和f两套区域特征，因此不能把它解释成最终预测模型完全没有使用f。同理，仅f聚类也仍测量D。这是用户确认的控制变量设计；要评价纯D预测与纯f预测，需要另做下游测量通道消融。', '',
        '不使用T2、ADC、D*、临床、直接全肿瘤影像组学；未重新纳入此前明确排除的迁移实验或20261010正式使用受限模型。', '',
        '## 随机病例与相互重叠', '',
        table(['组别', '100例阳性／阴性', '训练70例阳性／阴性', '测试30例阳性／阴性', '抽样种子', '划分种子'],
            [['SampleA', '68／32', '48／22', '20／10', 2026102001, 2026102011],
             ['SampleB', '68／32', '48／22', '20／10', 2026102002, 2026102012]]), '',
        f'两组均从既有138例中分层随机抽100例，组内不放回，名单在提取和建模前冻结，未按特征、预测或AUC挑病例。两组重叠{overlap["sample__sample"]}例，共同训练{overlap["train__train"]}例、共同测试{overlap["test__test"]}例；A训练与B测试重叠{overlap["train__test"]}例，A测试与B训练重叠{overlap["test__train"]}例。', '',
        '两组每个模型只使用本组70例训练，不跨组共享结局驱动筛选、预处理参数、模型权重或组合选择。逐患者无结局的影像计算可复用。组间病例重叠意味着这是两次相关的内部重复实验，不是两套独立外部验证；不能将两组预测拼接后当作独立200例或60例计算性能，也不按独立组计算AUC差异显著性。', '',
        '## 生境与特征计算', '',
        '沿用已有1 mm各向同性IVIM图像及有效掩膜：原生D＞0、0≤f＜0.7、NRMSE≤0.15；不增加D*特有质量条件。每个病例、每种原始／1 mm腐蚀／膨胀ROI单独重新聚类，所有组合共用相同掩膜。', '',
        '在当前有效ROI中对信号中心化／尺度化，再计算强度和3×3×3体素局部均值／总体SD；由于网格为1 mm，此处窗口为3 mm立方体，局部统计只按有效ROI内支持体素归一化。每维单独StandardScaler，使用相同最多4000个固定随机样本；选定维度进入KMeans，k＝3、n_init＝5、seed＝64212。未纳入维度不参与聚类距离或中心命名。', '',
        '保留两种逐患者标准化：P1/P99截尾后Z-score，以及中位数／1.4826MAD稳健Z-score（预设IQR、SD回退）。归一化参照固定原ROI；稳健Z的正仿射变化在当前ROI二次标准化中抵消，因此使用原规范信号来避免数值舍入成为额外分区干预。', '',
        '每个生境统一测量D／f的17项预先固定PyRadiomics指标和原定量／空间描述符，再用三生境同类指标均值、SD、范围及占比SD／最大值／熵形成156项标签不变区域汇总。此17项来自既有区域汇总定义，不在本轮按AUC挑选；直接特征类调用已逐项匹配原PyRadiomics完整接口，全六维也复现既有IVIM156项。', '',
        '训练前核验发现完整PyRadiomics的minimumROISize＝12实际要求纹理区域严格大于12体素。快速提取缓存中恰好12体素的生境在读取时统一将对应纹理汇总记为缺失，原定量／空间描述符仍按既有规则保留；原缓存未改写。该边界校正已用合成12体素区域与完整接口逐项核验，不依据结局或AUC，不删除病例或组合。', '',
        f'本轮纹理缺失／小生境错误记录共{error_n}条，见texture_errors.csv。缺失不按结局删除病例，进入训练折质量门槛及填补；各组合分区体素数、轮廓变体和有限特征数见partition_and_feature_audit.csv、feature_quality_summary.csv。', '',
        '## 相同下游模型与嵌套验证', '',
        '全部组合使用同一L2 liblinear逻辑回归家族。内层五折在Association／Stability排序、特征预算10／20／全部、C＝0.003／0.03／0.3／3中选优，每种标准化24个候选。预算是上限，另有全部选项；实际输入由训练数据的稳定性和去相关决定，没有固定限制为3或6项。', '',
        '每个训练折内独立执行ICC≥0.8、完整率≥90%、非常数门槛；Association用20次患者分层bootstrap关联稳定性，Stability按ICC排序；|Spearman|＜0.95去相关。训练中位数填补、P2.5/P97.5表格截尾及中位数/IQR缩放。若无合格特征则保留并明确标为不可用的患病率常数模型，不静默删组合或病例。', '',
        '内层目标沿用mean AUC − 0.10×折间SD − 0.30×max(mean拟合AUC−mean验证AUC−0.10,0)。每组63组合和两标准化使用相同外层5折seed42、内层62201—62205、最终70例内层62212；所有结局驱动步骤只用当前训练折。', '',
        '252个固定组合模型、6个训练内选择组合的别名模型及各自训练OOF Youden阈值全部核验并冻结后，统一进入一次测试阶段。固定组合每个模型只预测一次本组30例；别名直接共享既有概率。测试后没有重新拟合或更改阈值。', '',
        '训练拟合AUC来自最终70例模型预测自身训练病例；固定组合外层均值评估该组合的内层参数选择流程；汇总OOF将每个病例的外层未见预测合并。最终内层AUC参与了选择，不能代替外层验证。', '',
        '## 全63种组合：两组并列结果', '',
        '下表保留全部组合。均值只是两次相关实验的描述性平均；不作为独立重复次数或可直接引用的正式性能。图中缩写与上述定义一致。', '']
    for norm in study.NORMS:
        rows = summary[summary.Normalization == norm]
        body += ['### '+NORMS[norm], '', table(['组合', '聚类输入', 'A拟合', 'A外层均值', 'A测试', 'B拟合', 'B外层均值', 'B测试'],
            [[r.ComboID, r.ShortName, *[f'{r[group+"_"+field]:.4f}' for group in study.GROUPS for field in ('TrainFitAUC', 'OuterMeanAUC', 'AUC')]] for _, r in rows.iterrows()]), '']
    body += ['## 训练验证较高的固定组合：探索性排序', '',
        '下列按两组固定组合外层均值的描述性平均排序，未使用测试AUC排序。由于在63项里排名，这个固定组合外层最大值本身也有选择乐观性；评价完整选择流程应看开头的训练内选组合外层结果。', '']
    for norm in study.NORMS:
        top = summary[summary.Normalization == norm].sort_values(['MeanOuterAUC', 'DimensionCount', 'ComboID'], ascending=[False, True, True]).head(10)
        body += [NORMS[norm]+'：', '', table(['组合', '输入', 'A／B外层均值', 'A／B测试', '测试差绝对值'],
            [[r.ComboID, r.ShortName, f'{r.SampleA_OuterMeanAUC:.4f}／{r.SampleB_OuterMeanAUC:.4f}',
                f'{r.SampleA_AUC:.4f}／{r.SampleB_AUC:.4f}', f'{r.AbsoluteTestDifference:.4f}'] for _, r in top.iterrows()]), '']
    body += ['## 全六维参考及测试最高值', '',
        '以下最高测试值只用来完整披露筛查结果，不作为已经验证的最终推荐或论文性能。对126个标准化／组合进行测试筛查会放大最高点估计的乐观性。', '',
        table(['组别', '标准化', '全六维C63测试AUC（95%区间）', '最高测试组合', '最高测试AUC', '该组合外层均值'],
            [[group, NORMS[norm], metrics_ci(metrics.loc[(group, norm+'__C63')]), row.ComboID, f'{row.AUC:.4f}', f'{row.OuterMeanAUC:.4f}']
             for group in study.GROUPS for norm in study.NORMS for row in [merged[(merged.Group == group)&(merged.Normalization == norm)].sort_values(['AUC', 'ComboID'], ascending=[False, True]).iloc[0]]]), '',
        '每个固定组合相对同标准化C63的配对AUC差值及95%区间见paired_combo_vs_all6_auc.csv。同病例使用相同5000次按结局分层的患者抽样权重，条件于冻结预测；未进行多重比较校正，区间未包含整个模型开发过程重训的不确定性。', '',
        '## 测试前选中模型的分类表现', '', table(['组别', '选择范围', '组合', '阈值', '敏感度', '特异度', '准确率', 'TP/FN/TN/FP'],
            [[group, key.replace('Adaptive__', ''), c.ComboID, f'{m.Threshold:.6f}', f'{m.Sensitivity:.1%}', f'{m.Specificity:.1%}',
                f'{m.Accuracy:.1%}', '/'.join(str(int(m[s])) for s in ('TP', 'FN', 'TN', 'FP'))]
             for (group, key), c in adaptive.iterrows() for m in [metrics.loc[(group, key)]]]), '',
        '## 如何据此决定下一轮组合', '',
        '优先查看测试前由训练内锁定的组合及其完整流程外层验证；固定组合表用于判断结果是否在两组中方向一致，是否存在拟合高但外层低，或测试高但训练验证低。若两组训练内选出的组合不同，不把测试最高者自动定为正式模型。可将重复出现且训练验证较稳定的少数方案作为下一轮预先固定候选。', '',
        '当前138例来自同医院同队列，已多次探索。两组本轮训练都隔离各自30例测试，但重新抽样、以及在63项测试结果中挑选，不能清除历史开发暴露。测试结果可用于本轮探索判断；一旦据此决定下一轮方案，仍需新病例验证该选择的性能。当前不自动替换正式模型。', '',
        '## 核验与交付文件', '',
        '抽样及全部70个交叉验证折独立重放；六维全组合复现历史156项，快速17项纹理计算匹配完整PyRadiomics接口，未纳入维度的数值改变不影响聚类。所有候选内层拟合和验证AUC按病例排名独立重算；外层标签翻转、验证数值扰动、最终重拟合、OOF与阈值重放均在测试前完成。', '',
        '252个固定模型、36个内层组合选择记录和6个最终别名分别核验；7740条测试概率按独立sigmoid核验，258行指标及252项相对C63的配对差值区间独立重算。历史文件、模型与阈值哈希保持不变。', '',
        '主要文件：all63_side_by_side_summary.csv（126行并列汇总）、all63_two_samples_detailed_comparison.csv（252行详情）、adaptive_training_choices.csv、test_metrics.csv、paired_combo_vs_all6_auc.csv、combinations63.csv、all138_group_membership.csv；模型和逐折候选信息位于training目录，核验及交付清单见JSON文件。', '']
    for file in files:
        body += ['!['+file+']('+str((OUT/'figures'/file).resolve()).replace('\\', '/')+')', '']
    REPORT.write_text('\n'.join(body), encoding='utf-8')
    study.save(OUT/'report_generation.json', {'passed': True, 'report': REPORT.name, 'sha256': study.digest(REPORT),
        'all_fixed_result_rows': 252, 'all_comparison_rows': 126, 'figure_count': len(files),
        'model_predict_calls': 0, 'formal_model_use': False})
    print('63-way report completed;pretestselected references:')
    print(adaptive[['ComboID', 'Normalization', 'TrainFitAUC', 'OuterMeanAUC', 'PooledOOFAUC']].to_string())


if __name__ == '__main__':
    main()
