import i18n from "../../i18n";

/* Los hooks y la capa de api no son componentes, asi que no pueden usar
   useTranslation(): traducen con la instancia de i18next. El texto se resuelve
   al construir el error, de modo que un cambio de idioma posterior no vuelve a
   traducir un mensaje ya mostrado; se retraduce en la siguiente accion. */

export function translate(key: string): string {
  return i18n.t(key);
}

// El mensaje de un Error ya viene resuelto (suele ser el "detail" del backend);
// solo se traduce el fallback para lo que no es un Error.
export function errorMessage(error: unknown, fallbackKey: string): string {
  return error instanceof Error ? error.message : translate(fallbackKey);
}
