"""
pipeline_utils.py
==================
sklearn 기반 분류 모델 파이프라인 조립 / 학습 / 평가 / 비교 / 튜닝 유틸리티

구성 함수
---------
1. pipe()                      : 전처리 + 모델 파이프라인 조립
2. fit_and_predict()           : 학습 + 예측 (train/test 모두)
3. compare_models_with_timing() : 여러 모델의 학습/예측 소요 시간 비교
4. train_pred_test_reports()   : 성능 리포트 + 시각화 (confusion matrix, feature importance)
5. tune_model()                : GridSearchCV 기반 하이퍼파라미터 튜닝
6. compare_models()            : 여러 모델을 학습/평가하고 confusion matrix,
                                  feature importance, 정오답 공통점까지 리포트
7. analyze_correct_vs_incorrect() : 정답/오답 그룹 간 피처값 차이 분석
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


################################################################################
# 1. 파이프라인 조립 함수
################################################################################
def pipe(model, num_f: list, cat_f: list, X: pd.DataFrame) -> Pipeline:
    """수치형/범주형 전처리와 모델을 하나의 Pipeline으로 조립한다.

    Parameters
    ----------
    model : sklearn 호환 추정기 (분류/회귀 모델)
    num_f : 수치형 컬럼명 리스트
    cat_f : 범주형 컬럼명 리스트
    X     : 컬럼 검증에 사용할 학습 데이터 (pandas DataFrame)

    Returns
    -------
    Pipeline : 'preprocessor' + 'estimator' 스텝으로 구성된 파이프라인
    """
    if not num_f and not cat_f:
        raise ValueError("num_f와 cat_f 중 최소 하나는 비어있으면 안 됩니다.")
    if model is None:
        raise ValueError("model 값이 비어있으면 안 됩니다.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X는 pandas DataFrame이어야 합니다.")
    if isinstance(num_f, str) or isinstance(cat_f, str):
        raise TypeError("num_f와 cat_f는 컬럼명의 리스트여야 합니다.")

    overlap = set(num_f) & set(cat_f)
    if overlap:
        raise ValueError(f"num_f와 cat_f에 중복된 컬럼이 있습니다: {overlap}")

    missing_cols = (set(num_f) | set(cat_f)) - set(X.columns)
    if missing_cols:
        raise ValueError(f"X에 존재하지 않는 컬럼입니다: {missing_cols}")

    # num_f가 실제 숫자형인지 확인 (bool은 제외)
    non_numeric = [
        col for col in num_f
        if not is_numeric_dtype(X[col]) or is_bool_dtype(X[col])
    ]
    if non_numeric:
        raise TypeError(f"num_f에 숫자형이 아닌 컬럼이 있습니다: {non_numeric}")

    num_transformer = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_transformer', num_transformer, num_f),
            ('cat_transformer', cat_transformer, cat_f)
        ],
        remainder='drop'
    )
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', model)
    ])

    return model_pipeline


################################################################################
# 2. 학습 및 예측 함수
################################################################################
def fit_and_predict(pipeline: Pipeline, X_train, y_train, X_test):
    """조립된 파이프라인으로 학습 후 train/test 예측 결과를 반환한다.

    estimator가 predict_proba를 지원하지 않으면 probs는 None으로 반환된다.
    """
    pipeline.fit(X_train, y_train)

    estimator = pipeline.named_steps['estimator']
    supports_proba = hasattr(estimator, 'predict_proba')

    preds = pipeline.predict(X_test)
    train_preds = pipeline.predict(X_train)

    if supports_proba:
        probs = pipeline.predict_proba(X_test)[:, 1]
        train_probs = pipeline.predict_proba(X_train)[:, 1]
    else:
        probs, train_probs = None, None

    return pipeline, preds, probs, train_preds, train_probs


################################################################################
# 3. 학습/예측 시간 비교 함수
################################################################################
def compare_models_with_timing(models: dict, num_f, cat_f, X_train, y_train, X_test, y_test):
    """여러 모델의 fit/predict 소요 시간을 비교한다."""
    results = []
    for name, model in models.items():
        pipeline = pipe(model=model, num_f=num_f, cat_f=cat_f, X=X_train)

        start = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start

        start = time.time()
        pipeline.predict(X_test)
        predict_time = time.time() - start

        results.append({
            'model': name,
            'fit_time_sec': fit_time,
            'predict_time_sec': predict_time,
        })
    return pd.DataFrame(results)


################################################################################
# 4. 리포트 및 시각화 함수
################################################################################
def plot_feature_importance(model_pipeline: Pipeline, top_n: int = 20):
    """estimator 종류에 상관없이 feature importance / 계수를 시각화한다.

    tree 계열(feature_importances_), 선형 계열(coef_) 모두 지원.
    둘 다 없는 모델은 안내 메시지만 출력하고 종료한다.
    """
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    estimator = model_pipeline.named_steps['estimator']

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importances = estimator.coef_.ravel()
    else:
        print(f"[안내] {type(estimator).__name__}은(는) feature importance 시각화를 지원하지 않습니다.")
        return

    imp_df = (
        pd.DataFrame({'feature': feature_names, 'importance': importances})
        .assign(abs_importance=lambda d: d['importance'].abs())
        .sort_values('abs_importance', ascending=False)
        .head(top_n)
        .sort_values('importance')
    )

    plt.figure(figsize=(8, max(4, len(imp_df) * 0.3)))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.title(f"Feature Importance ({type(estimator).__name__})")
    plt.tight_layout()
    plt.show()


def train_pred_test_reports(
    model_pipeline: Pipeline,
    X_train, y_train, X_test, y_test,
    preds, probs, train_preds, train_probs,
    cv: int = 5,
):
    """학습/테스트 성능, cross-validation 점수, confusion matrix, feature importance를
    한 번에 출력/시각화한다.

    probs가 None인 경우(predict_proba 미지원 모델) ROC AUC 관련 출력은 건너뛴다.
    """
    print("=" * 60)
    print(f"Estimator: {type(model_pipeline.named_steps['estimator']).__name__}")
    print("=" * 60)

    # ---- 과적합 확인 ----
    print("\n[Overfitting Check]")
    if probs is not None and train_probs is not None:
        print(f"Train ROC AUC : {roc_auc_score(y_train, train_probs):.4f}")
        print(f"Test  ROC AUC : {roc_auc_score(y_test, probs):.4f}")
    else:
        print("(predict_proba 미지원 모델 - ROC AUC 생략)")
    print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"Test  Accuracy: {accuracy_score(y_test, preds):.4f}")

    # ---- Cross-validation ----
    print("\n[Cross-Validation]")
    scoring = 'roc_auc' if probs is not None else 'accuracy'
    cv_result = cross_validate(
        model_pipeline, X_train, y_train, cv=cv, scoring=scoring
    )
    print(f"CV {scoring} mean: {cv_result['test_score'].mean():.4f} "
          f"(+/- {cv_result['test_score'].std():.4f})")
    print(f"Avg fit time  : {cv_result['fit_time'].mean():.4f}s "
          f"(+/- {cv_result['fit_time'].std():.4f})")
    print(f"Avg score time: {cv_result['score_time'].mean():.4f}s "
          f"(+/- {cv_result['score_time'].std():.4f})")

    # ---- Feature Importance ----
    print("\n[Feature Importance]")
    plot_feature_importance(model_pipeline)

    # ---- Confusion Matrix ----
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # ---- Classification Report ----
    print("\n[Classification Report]")
    print(classification_report(y_test, preds, digits=4))

    return {
        'train_accuracy': accuracy_score(y_train, train_preds),
        'test_accuracy': accuracy_score(y_test, preds),
        'cv_score_mean': cv_result['test_score'].mean(),
        'cv_score_std': cv_result['test_score'].std(),
    }


################################################################################
# 5. 하이퍼파라미터 튜닝 함수
################################################################################
def tune_model(model, param_grid: dict, num_f, cat_f, X, y, cv: int = 5, scoring: str = 'f1'):
    """GridSearchCV로 파이프라인 내 estimator 파라미터를 튜닝한다.

    param_grid의 키는 'estimator__' 접두어를 붙여야 한다.
    예: {'estimator__n_estimators': [100, 300], 'estimator__max_depth': [3, 5]}
    """
    pipeline = pipe(model=model, num_f=num_f, cat_f=cat_f, X=X)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


################################################################################
# 6. 정답/오답 공통점 분석 함수
################################################################################
def analyze_correct_vs_incorrect(X_test: pd.DataFrame, y_test, preds, num_f, cat_f, top_n: int = 5):
    """정답(correct)과 오답(incorrect) 그룹 간 피처값 차이가 큰 순으로 정리한다.

    - 수치형: 그룹별 평균과 그 차이(절대값) 기준 정렬
    - 범주형: 그룹별 최빈값과 그 비중(proportion)

    Returns
    -------
    dict : {'numeric': DataFrame, 'categorical': DataFrame} (해당 피처가 없으면 키 생략)
    """
    df = X_test.reset_index(drop=True).copy()
    y_true = pd.Series(y_test).reset_index(drop=True)
    preds_s = pd.Series(preds).reset_index(drop=True)
    df['_correct'] = (preds_s.values == y_true.values)

    result = {}

    if num_f:
        num_summary = df.groupby('_correct')[list(num_f)].mean().T
        num_summary = num_summary.rename(columns={True: 'correct_mean', False: 'incorrect_mean'})
        for col in ('correct_mean', 'incorrect_mean'):
            if col not in num_summary.columns:
                num_summary[col] = float('nan')
        num_summary['abs_diff'] = (num_summary['correct_mean'] - num_summary['incorrect_mean']).abs()
        result['numeric'] = num_summary.sort_values('abs_diff', ascending=False).head(top_n)

    if cat_f:
        rows = []
        for col in cat_f:
            for grp_val, label in ((True, 'correct'), (False, 'incorrect')):
                subset = df.loc[df['_correct'] == grp_val, col]
                if subset.empty:
                    continue
                mode_val = subset.mode().iloc[0]
                proportion = (subset == mode_val).mean()
                rows.append({
                    'feature': col, 'group': label,
                    'most_common_value': mode_val, 'proportion': round(proportion, 4)
                })
        result['categorical'] = pd.DataFrame(rows)

    return result


def plot_confusion_matrix_for(model_pipeline: Pipeline, X_test, y_test, preds, title: str = ""):
    """단일 모델의 confusion matrix를 시각화한다."""
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {title}" if title else "Confusion Matrix")
    plt.show()


################################################################################
# 7. 여러 모델 비교 함수 (성능 + confusion matrix + feature importance + 정오답 분석)
################################################################################
def compare_models(
    models: dict, num_f, cat_f, X_train, y_train, X_test, y_test,
    cv: int = 5,
    scoring=('accuracy', 'f1', 'roc_auc'),
    show_confusion_matrix: bool = True,
    show_feature_importance: bool = True,
    show_correct_incorrect_analysis: bool = True,
    top_n_diff: int = 5,
):
    """models: {'모델이름': model_object} 형태의 딕셔너리를 학습/평가/비교한다.

    각 모델에 대해 순차적으로:
      1) X_train으로 cross-validation 점수 산출 (scoring 지표별 mean/std)
      2) X_train으로 fit 후 X_test에 대해 predict
      3) confusion matrix 시각화
      4) feature importance 시각화 (지원하는 모델만)
      5) 정답/오답 그룹 간 피처값 차이 top_n_diff개 분석

    Returns
    -------
    summary_df : 모델별 성능 지표 DataFrame (score 기준 내림차순 정렬)
    details    : {'모델이름': {'pipeline':..., 'preds':..., 'confusion_matrix':...,
                              'correct_incorrect_analysis': {...}}} 형태의 상세 결과 딕셔너리
    """
    scoring = list(scoring)
    summary_rows = []
    details = {}

    for name, model in models.items():
        print("=" * 60)
        print(f"[{name}]")
        print("=" * 60)

        pipeline = pipe(model=model, num_f=num_f, cat_f=cat_f, X=X_train)

        # 1) cross-validation 점수
        cv_result = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
        )
        row = {'model': name}
        for metric in scoring:
            row[f'{metric}_mean'] = cv_result[f'test_{metric}'].mean()
            row[f'{metric}_std'] = cv_result[f'test_{metric}'].std()
        summary_rows.append(row)

        # 2) 학습 + 테스트 예측
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        model_detail = {'pipeline': pipeline, 'preds': preds}

        # 3) confusion matrix
        if show_confusion_matrix:
            plot_confusion_matrix_for(pipeline, X_test, y_test, preds, title=name)
            model_detail['confusion_matrix'] = confusion_matrix(y_test, preds)

        # 4) feature importance
        if show_feature_importance:
            plot_feature_importance(pipeline)

        # 5) 정답/오답 공통점 분석
        if show_correct_incorrect_analysis:
            analysis = analyze_correct_vs_incorrect(
                X_test, y_test, preds, num_f, cat_f, top_n=top_n_diff
            )
            if 'numeric' in analysis:
                print(f"\n[{name}] 정답/오답 그룹 간 수치형 피처 평균 차이 (상위 {top_n_diff}개)")
                print(analysis['numeric'])
            if 'categorical' in analysis:
                print(f"\n[{name}] 정답/오답 그룹 간 범주형 피처 최빈값")
                print(analysis['categorical'])
            model_detail['correct_incorrect_analysis'] = analysis

        details[name] = model_detail
        print()

    summary_df = pd.DataFrame(summary_rows).sort_values(by=f'{scoring[0]}_mean', ascending=False)
    return summary_df, details
