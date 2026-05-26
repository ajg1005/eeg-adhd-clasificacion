import PropTypes from "prop-types";

export const datasetStatsShape = PropTypes.shape({
  class_distribution: PropTypes.objectOf(PropTypes.number).isRequired,
  columns: PropTypes.number.isRequired,
  eeg_columns: PropTypes.arrayOf(PropTypes.string).isRequired,
  missing_required_columns: PropTypes.arrayOf(PropTypes.string).isRequired,
  n_patients: PropTypes.number.isRequired,
  patients: PropTypes.arrayOf(
    PropTypes.shape({
      class_label: PropTypes.string.isRequired,
      patient_id: PropTypes.string.isRequired,
      rows: PropTypes.number.isRequired,
    }),
  ).isRequired,
  rows: PropTypes.number.isRequired,
});

export const fileShape = PropTypes.shape({
  name: PropTypes.string.isRequired,
  size: PropTypes.number,
});

export const modelInfoShape = PropTypes.shape({
  channels: PropTypes.arrayOf(PropTypes.string),
  display_name: PropTypes.string,
  epoch_size: PropTypes.number,
  feature_mode: PropTypes.string,
  model_family: PropTypes.string,
  model_name: PropTypes.string,
  n_features: PropTypes.number,
  sfreq: PropTypes.number,
  step_size: PropTypes.number,
});

export const modelOptionShape = PropTypes.shape({
  default_params: PropTypes.object,
  display_name: PropTypes.string.isRequired,
  parameters: PropTypes.object,
});

export const patientResultShape = PropTypes.shape({
  adhd_epoch_percentage: PropTypes.number.isRequired,
  control_epoch_percentage: PropTypes.number.isRequired,
  correct: PropTypes.bool.isRequired,
  n_epochs: PropTypes.number.isRequired,
  patient_id: PropTypes.string.isRequired,
  predicted_label: PropTypes.string.isRequired,
  true_label: PropTypes.string.isRequired,
});
