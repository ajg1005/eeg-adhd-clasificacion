import { useTranslation } from "react-i18next";

interface ViewHeadingProps {
  ledeKey: string;
  titleKey: string;
}
export function ViewHeading({ ledeKey, titleKey }: ViewHeadingProps) {
  const { t } = useTranslation();

  return (
    <div className="view-heading">
      <h1>{t(titleKey)}</h1>
      <p className="view-lede">{t(ledeKey)}</p>
    </div>
  );
}
