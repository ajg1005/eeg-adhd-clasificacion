import { useTranslation } from "react-i18next";

interface ViewHeadingProps {
  ledeKey: string;
  titleKey: string;
}

// Titulo de cada pestaña. Vive en la vista y no en la cabecera global, que se
// queda como una barra fija de 64px con la navegacion.
export function ViewHeading({ ledeKey, titleKey }: ViewHeadingProps) {
  const { t } = useTranslation();

  return (
    <div className="view-heading">
      <h1>{t(titleKey)}</h1>
      <p className="view-lede">{t(ledeKey)}</p>
    </div>
  );
}
