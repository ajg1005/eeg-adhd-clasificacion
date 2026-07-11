import PropTypes from "prop-types";

export function ModelSelectField({ disabled, label, onChange, options, value }) {
  return (
    <label>
      {label}
      <select disabled={disabled} onChange={onChange} value={value}>
        {options.map((option) => (
          <option
            disabled={option.disabled}
            key={option.value}
            value={option.value}
          >
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

ModelSelectField.propTypes = {
  disabled: PropTypes.bool,
  label: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  options: PropTypes.arrayOf(
    PropTypes.shape({
      disabled: PropTypes.bool,
      label: PropTypes.string.isRequired,
      value: PropTypes.string.isRequired,
    }),
  ).isRequired,
  value: PropTypes.string.isRequired,
};

ModelSelectField.defaultProps = {
  disabled: false,
};
