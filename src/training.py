import numpy as np
import polars as pl


def prepare_features_for_classification(
    df: pl.DataFrame, 
    log_columns: list[str], 
    asinh_columns: list[str], 
    drop_columns: list[str],
    eps: float = 1e-18
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = df.with_columns(
        *[pl.col(col).log10() for col in [log_columns]],
        *[pl.col(col).arcsinh() for col in [asinh_columns]]
    )
    dropped_columns = df.select(['path'] + drop_columns)
    return df.drop(drop_columns), dropped_columns

def make_predictions(clf, label_encoder, X_preprocessed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #class_mapper = {k: v for k, v in enumerate(clf.classes_)}
    #inv_mapper = {v: k for k, v in class_mapper.items()}    
    y_pred_proba = clf.predict_proba(X_preprocessed)    
    y_pred = label_encoder.inverse_transform(y_pred_proba.argmax(axis=1))
    #y_pred = np.array([class_mapper[pred] for pred in y_pred])
    y_entropy = -np.sum(y_pred_proba*np.log(y_pred_proba+1e-12), axis=1)/np.log(y_pred_proba.shape[1])
    return y_pred_proba, y_pred, y_entropy

def prepare_predictions_dataframe(clf, label_encoder, X_preprocessed, paths, drop_nans: bool = True) -> pl.DataFrame:
    y_pred_proba, y_pred, y_entropy = make_predictions(clf, label_encoder, X_preprocessed)
    non_nan_mask = ~np.isnan(y_entropy)
    class_names = label_encoder.classes_
    df_pred = pl.DataFrame({
        'path': paths[non_nan_mask]
    } | {
        name: y_pred_proba[non_nan_mask, k] for k, name in enumerate(class_names)
    } | {
        'best_rf_class': y_pred[non_nan_mask], 'rf_entropy': y_entropy
    })
    df_pred = df_pred.with_columns(
        pl.concat_list([pl.col(k) for k in class_names]).list.max().alias('best_rf_proba')
    )
    return df_pred