import type { ChangeEventHandler } from "react";

import type { SelectOption } from "../types";

interface ModelSelectFieldProps {
  disabled?: boolean;
  label: string;
  onChange: ChangeEventHandler<HTMLSelectElement>;
  options: SelectOption[];
  value: string;
}

export function ModelSelectField({
  disabled = false,
  label,
  onChange,
  options,
  value,
}: ModelSelectFieldProps) {
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
