"use client";


import {
  useEffect,
  useState,
} from "react";


import {
  archivePreparationSession,
  deletePreparationSession,
  listPreparationSessions,
  renamePreparationSession,
  restorePreparationSession,
} from "./preparationApi";


import type {
  PreparationSessionCatalogItem,
  PreparationSessionCatalogResponse,
  PreparationSessionView,
} from "./preparationTypes";


import styles from "./PreparationWorkflowHistoryPanel.module.css";


/* ============================================================
   WORKFLOW HISTORY
   PREPARATION_WORKFLOW_HISTORY_FRONTEND_V0_1

   WORKFLOW METADATA
   PREPARATION_WORKFLOW_METADATA_FRONTEND_V0_1
============================================================ */


type PreparationWorkflowHistoryPanelProps = {
  activeWorkflowId:
    string |
    null;

  /*
   * PREPARATION_WORKFLOW_ARCHIVE_LOAD_GUARD_V0_2
   *
   * True only when the currently mounted workflow has finished
   * the data-loading lifecycle required before it may be archived.
   */
  activeWorkflowCanArchive:
    boolean;

  onOpenWorkflow:
    (
      workflowId:
        string
    ) => void;

  onArchiveActiveWorkflow:
    () => void;
};


const STAGE_LABELS:
  Record<
    string,
    string
  > = {
    import:
      "Import",

    understand:
      "Compr\u00e9hension",

    quality:
      "Qualit\u00e9",

    clean:
      "Nettoyage",

    transform:
      "Transformation",

    combine:
      "Combinaison",

    validate:
      "Validation",
  };


function statusLabel(
  session:
    PreparationSessionView
): string {
  if (
    session.snapshot
      .ready_for_analysis
  ) {
    return (
      "Pr\u00eat pour analyse"
    );
  }


  const nextStage =
    session.snapshot
      .next_stage;


  if (
    nextStage
  ) {
    return (
      `Prochaine \u00e9tape : ${
        STAGE_LABELS[
          nextStage
        ] ??
        nextStage
      }`
    );
  }


  return (
    "Pr\u00e9paration en cours"
  );
}


function formatDate(
  value:
    string
): string {
  const date =
    new Date(
      value
    );


  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return value;
  }


  return (
    new Intl.DateTimeFormat(
      "fr-FR",
      {
        dateStyle:
          "medium",

        timeStyle:
          "short",
      }
    )
    .format(
      date
    )
  );
}


function shortId(
  workflowId:
    string
): string {
  if (
    workflowId.length <=
    28
  ) {
    return workflowId;
  }


  return (
    `${
      workflowId.slice(
        0,
        12
      )
    }...${
      workflowId.slice(
        -8
      )
    }`
  );
}


export default function PreparationWorkflowHistoryPanel({
  activeWorkflowId,
  activeWorkflowCanArchive,
  onOpenWorkflow,
  onArchiveActiveWorkflow,
}: PreparationWorkflowHistoryPanelProps) {
  const [
    catalog,
    setCatalog,
  ] =
    useState<
      PreparationSessionCatalogResponse |
      null
    >(
      null
    );


  const [
    loading,
    setLoading,
  ] =
    useState(
      true
    );


  const [
    error,
    setError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    refreshToken,
    setRefreshToken,
  ] =
    useState(
      0
    );


  const [
    busyWorkflowId,
    setBusyWorkflowId,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    editingWorkflowId,
    setEditingWorkflowId,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    draftName,
    setDraftName,
  ] =
    useState(
      ""
    );


  useEffect(
    () => {
      const controller =
        new AbortController();


      setLoading(
        true
      );

      setError(
        null
      );


      void (
        async () => {
          try {
            const response =
              await listPreparationSessions(
                controller.signal
              );


            if (
              controller.signal
                .aborted
            ) {
              return;
            }


            setCatalog(
              response
            );
          } catch (
            caughtError
          ) {
            if (
              controller.signal
                .aborted
            ) {
              return;
            }


            setError(
              caughtError instanceof
                Error
                ? caughtError.message
                : (
                    "Impossible de charger "
                    +
                    "l\u2019historique."
                  )
            );
          } finally {
            if (
              !controller.signal
                .aborted
            ) {
              setLoading(
                false
              );
            }
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      activeWorkflowId,
      refreshToken,
    ]
  );


  const sessions =
    catalog?.sessions ??
    [];


  // ==========================================================
  // PERMANENT WORKFLOW DELETE
  // PREPARATION_WORKFLOW_PERMANENT_DELETE_FRONTEND_V0_1
  // ==========================================================


  const [
    deleteBusyWorkflowId,
    setDeleteBusyWorkflowId,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  async function handleDeleteArchivedWorkflow(
    item:
      PreparationSessionCatalogResponse[
        "sessions"
      ][number]
  ) {
    const workflowId =
      item.session.workflow_id;


    /*
     * Browser visibility is not the security boundary.
     *
     * The backend independently validates archive state,
     * workflow identity, current name and revision.
     */
    if (
      !item.archived
    ) {
      setError(
        "Le workflow doit être archivé avant sa suppression définitive."
      );

      return;
    }


    const confirmation =
      window.prompt(
        [
          "Suppression définitive du workflow.",
          "",
          "Cette action est irréversible.",
          "",
          "Tapez exactement le nom suivant pour confirmer :",
          item.display_name,
        ].join(
          "\n"
        )
      );


    if (
      confirmation ===
      null
    ) {
      return;
    }


    if (
      confirmation !==
      item.display_name
    ) {
      setError(
        "Suppression annulée : le nom saisi ne correspond pas exactement au workflow."
      );

      return;
    }


    setDeleteBusyWorkflowId(
      workflowId
    );

    setError(
      null
    );


    try {
      await deletePreparationSession(
        workflowId,
        item.display_name,
        item.session.revision
      );


      /*
       * Do not hide the deleted workflow optimistically.
       *
       * Reload the authoritative server catalog and let the
       * PreparationSession root prove that it disappeared.
       */
      /*
       * PREPARATION_WORKFLOW_DELETE_ARCHIVE_AUTHORITY_V0_1
       *
       * The server-owned archive state is the authority for
       * whether deletion is allowed.
       *
       * If WorkspaceClient still has this workflow mounted,
       * detach the browser only AFTER the backend confirms the
       * permanent deletion.
       */
      if (
        workflowId ===
        activeWorkflowId
      ) {
        onArchiveActiveWorkflow?.();
      }


      const refreshedCatalog =
        await listPreparationSessions();


      setCatalog(
        refreshedCatalog
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : (
              "La suppression définitive du workflow a échoué."
            )
      );
    } finally {
      setDeleteBusyWorkflowId(
        null
      );
    }
  }


  const activeSessions =
    sessions.filter(
      (
        item
      ) =>
        !item.archived
    );


  const archivedSessions =
    sessions.filter(
      (
        item
      ) =>
        item.archived
    );


  function refresh() {
    setRefreshToken(
      (
        current
      ) =>
        current + 1
    );
  }


  function beginRename(
    item:
      PreparationSessionCatalogItem
  ) {
    setEditingWorkflowId(
      item.session
        .workflow_id
    );

    setDraftName(
      item.display_name
    );

    setError(
      null
    );
  }


  function cancelRename() {
    setEditingWorkflowId(
      null
    );

    setDraftName(
      ""
    );
  }


  async function saveRename(
    item:
      PreparationSessionCatalogItem
  ) {
    const workflowId =
      item.session
        .workflow_id;

    const name =
      draftName.trim();


    if (
      !name
    ) {
      setError(
        "Le nom du workflow ne peut pas \u00eatre vide."
      );

      return;
    }


    setBusyWorkflowId(
      workflowId
    );

    setError(
      null
    );


    try {
      await renamePreparationSession(
        workflowId,
        name
      );

      cancelRename();

      refresh();
    } catch (
      caughtError
    ) {
      setError(
        caughtError instanceof
          Error
          ? caughtError.message
          : "Impossible de renommer le workflow."
      );
    } finally {
      setBusyWorkflowId(
        null
      );
    }
  }


  async function archiveWorkflow(
    item:
      PreparationSessionCatalogItem
  ) {
    const workflowId =
      item.session
        .workflow_id;


    /*
     * PREPARATION_WORKFLOW_ARCHIVE_LOAD_GUARD_V0_2
     *
     * The UI disables the button during data loading.
     * This second check makes the handler itself fail closed
     * if invoked from stale browser state.
     */
    if (
      workflowId ===
        activeWorkflowId &&
      !activeWorkflowCanArchive
    ) {
      setError(
        "Attendez la fin du chargement des données avant d’archiver ce workflow."
      );

      return;
    }


    setBusyWorkflowId(
      workflowId
    );

    setError(
      null
    );


    try {
      await archivePreparationSession(
        workflowId
      );

      cancelRename();


      if (
        workflowId ===
        activeWorkflowId
      ) {
        onArchiveActiveWorkflow();
      }


      refresh();
    } catch (
      caughtError
    ) {
      setError(
        caughtError instanceof
          Error
          ? caughtError.message
          : "Impossible d\u2019archiver le workflow."
      );
    } finally {
      setBusyWorkflowId(
        null
      );
    }
  }


  async function restoreWorkflow(
    item:
      PreparationSessionCatalogItem
  ) {
    const workflowId =
      item.session
        .workflow_id;


    setBusyWorkflowId(
      workflowId
    );

    setError(
      null
    );


    try {
      await restorePreparationSession(
        workflowId
      );

      refresh();
    } catch (
      caughtError
    ) {
      setError(
        caughtError instanceof
          Error
          ? caughtError.message
          : "Impossible de restaurer le workflow."
      );
    } finally {
      setBusyWorkflowId(
        null
      );
    }
  }


  function renderItem(
    item:
      PreparationSessionCatalogItem
  ) {
    const session =
      item.session;

    const workflowId =
      session.workflow_id;

    const isActive =
      workflowId ===
      activeWorkflowId;

    const isBusy =
      busyWorkflowId ===
      workflowId;

    const isEditing =
      editingWorkflowId ===
      workflowId;


    return (
      <article
        className={
          `${styles.item} ${
            isActive
              ? styles.itemActive
              : ""
          } ${
            item.archived
              ? styles.itemArchived
              : ""
          }`
        }
        key={
          workflowId
        }
      >
        <div
          className={
            styles.itemMain
          }
        >
          <div
            className={
              styles.itemHeading
            }
          >
            {
              isEditing
                ? (
                    <input
                      className={
                        styles.renameInput
                      }
                      value={
                        draftName
                      }
                      maxLength={
                        120
                      }
                      disabled={
                        isBusy
                      }
                      autoFocus
                      onChange={
                        (
                          event
                        ) =>
                          setDraftName(
                            event.target.value
                          )
                      }
                      onKeyDown={
                        (
                          event
                        ) => {
                          if (
                            event.key ===
                            "Enter"
                          ) {
                            event.preventDefault();

                            void saveRename(
                              item
                            );
                          }


                          if (
                            event.key ===
                            "Escape"
                          ) {
                            cancelRename();
                          }
                        }
                      }
                    />
                  )
                : (
                    <strong
                      className={
                        styles.workflowName
                      }
                    >
                      {
                        item.display_name
                      }
                    </strong>
                  )
            }


            {
              isActive
                ? (
                    <span
                      className={
                        styles.activeBadge
                      }
                    >
                      Actif
                    </span>
                  )
                : null
            }


            {
              item.archived
                ? (
                    <span
                      className={
                        styles.archivedBadge
                      }
                    >
                      Archiv\u00e9
                    </span>
                  )
                : null
            }


            {
              item.name_source ===
                "automatic"
                ? (
                    <span
                      className={
                        styles.automaticBadge
                      }
                    >
                      Nom automatique
                    </span>
                  )
                : null
            }
          </div>


          <div
            className={
              styles.workflowId
            }
            title={
              workflowId
            }
          >
            {
              shortId(
                workflowId
              )
            }
          </div>


          <div
            className={
              styles.meta
            }
          >
            <span>
              {
                session
                  .selected_analysis_dataset_ids
                  .length
              }
              {
                session
                  .selected_analysis_dataset_ids
                  .length >
                1
                  ? " datasets"
                  : " dataset"
              }
            </span>

            <span>
              {
                `R\u00e9vision ${
                  session.revision
                }`
              }
            </span>

            <span>
              {
                `Modifi\u00e9 ${
                  formatDate(
                    item.updated_at_utc
                  )
                }`
              }
            </span>
          </div>


          <div
            className={
              styles.statusRow
            }
          >
            <span
              className={
                styles.status
              }
            >
              {
                statusLabel(
                  session
                )
              }
            </span>

            <span
              className={
                styles.createdAt
              }
            >
              {
                `Cr\u00e9\u00e9 ${
                  formatDate(
                    item.created_at_utc
                  )
                }`
              }
            </span>
          </div>
        </div>


        <div
          className={
            styles.actions
          }
        >
          {
            item.archived
              ? (
                  <button
                    className={
                      styles.primaryButton
                    }
                    type="button"
                    disabled={
                      isBusy
                    }
                    onClick={
                      () => {
                        void restoreWorkflow(
                          item
                        );
                      }
                    }
                  >
                    {
                      isBusy
                        ? "Restauration..."
                        : "Restaurer"
                    }
                  </button>
                )
              : isEditing
                ? (
                    <>
                      <button
                        className={
                          styles.primaryButton
                        }
                        type="button"
                        disabled={
                          isBusy
                        }
                        onClick={
                          () => {
                            void saveRename(
                              item
                            );
                          }
                        }
                      >
                        Enregistrer
                      </button>

                      <button
                        className={
                          styles.secondaryButton
                        }
                        type="button"
                        disabled={
                          isBusy
                        }
                        onClick={
                          cancelRename
                        }
                      >
                        Annuler
                      </button>
                    </>
                  )
                : (
                    <>
                      <button
                        className={
                          styles.secondaryButton
                        }
                        type="button"
                        disabled={
                          isBusy
                        }
                        onClick={
                          () =>
                            beginRename(
                              item
                            )
                        }
                      >
                        Renommer
                      </button>

                      <button
                        className={
                          styles.archiveButton
                        }
                        type="button"
                        disabled={
                          isBusy ||
                          (
                            isActive &&
                            !activeWorkflowCanArchive
                          )
                        }
                        title={
                          (
                            isActive &&
                            !activeWorkflowCanArchive
                          )
                            ? (
                                "Le workflow pourra être archivé lorsque le chargement des données sera terminé."
                              )
                            : undefined
                        }
                        onClick={
                          () => {
                            void archiveWorkflow(
                              item
                            );
                          }
                        }
                      >
                        {
                          (
                            isActive &&
                            !activeWorkflowCanArchive
                          )
                            ? (
                                "Chargement des données…"
                              )
                            : (
                                isBusy
                                  ? "Archivage…"
                                  : "Archiver"
                              )
                        }
                      </button>

                      <button
                        className={
                          isActive
                            ? styles.openButtonActive
                            : styles.openButton
                        }
                        type="button"
                        disabled={
                          isActive ||
                          isBusy
                        }
                        onClick={
                          () =>
                            onOpenWorkflow(
                              workflowId
                            )
                        }
                      >
                        {
                          isActive
                            ? "Ouvert"
                            : "Ouvrir"
                        }
                      </button>
                    </>
                  )
          }
        </div>
      </article>
    );
  }


  return (
    <section
      className={
        styles.history
      }
    >
      <div
        className={
          styles.header
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Historique
          </span>

          <h3
            className={
              styles.title
            }
          >
            {
              "Workflows enregistr\u00e9s"
            }
          </h3>

          <p
            className={
              styles.description
            }
          >
            {
              "Retrouvez, renommez et archivez vos pr\u00e9parations conserv\u00e9es par DataLens."
            }
          </p>
        </div>


        <button
          className={
            styles.refreshButton
          }
          type="button"
          disabled={
            loading
          }
          onClick={
            refresh
          }
        >
          {
            loading
              ? "Actualisation..."
              : "Actualiser"
          }
        </button>
      </div>


      {
        error
          ? (
              <div
                className={
                  styles.error
                }
              >
                {
                  error
                }
              </div>
            )
          : null
      }


      {
        loading &&
        catalog ===
          null
          ? (
              <div
                className={
                  styles.loading
                }
              >
                {
                  "Chargement de l\u2019historique..."
                }
              </div>
            )
          : null
      }


      {
        activeSessions.length >
        0
          ? (
              <div
                className={
                  styles.section
                }
              >
                <div
                  className={
                    styles.sectionTitle
                  }
                >
                  Actifs
                  {" \u00b7 "}
                  {
                    activeSessions.length
                  }
                </div>

                <div
                  className={
                    styles.list
                  }
                >
                  {
                    activeSessions.map(
                      renderItem
                    )
                  }
                </div>
              </div>
            )
          : null
      }


      {
        archivedSessions.length >
        0
          ? (
              <div
                className={
                  styles.section
                }
              >
                <div
                  className={
                    styles.sectionTitle
                  }
                >
                  Archives
                  {" \u00b7 "}
                  {
                    archivedSessions.length
                  }
                </div>

                <div
                  className={
                    styles.list
                  }
                >
                  {
                    archivedSessions.map(
                      (
                        item
                      ) => (
                        <div
                          className={
                            styles.archivedItem
                          }
                          key={
                            `archived:${
                              item.session.workflow_id
                            }`
                          }
                        >
                          {
                            renderItem(
                              item
                            )
                          }


                          <div
                            className={
                              styles.deleteRow
                            }
                          >
                            {/*
                              PREPARATION_WORKFLOW_DELETE_ARCHIVE_AUTHORITY_V0_1

                              This button is rendered only inside
                              archivedSessions.

                              The backend independently verifies the
                              archived state before permanent deletion.
                            */}
                            <button
                              className={
                                styles.deleteButton
                              }
                              type="button"
                              disabled={
                                deleteBusyWorkflowId !==
                                  null
                              }
                              onClick={
                                () => {
                                  void handleDeleteArchivedWorkflow(
                                    item
                                  );
                                }
                              }
                            >
                              {
                                deleteBusyWorkflowId ===
                                  item.session.workflow_id
                                  ? (
                                      "Suppression…"
                                    )
                                  : (
                                      "Supprimer définitivement"
                                    )
                              }
                            </button>
                          </div>
                        </div>
                      )
                    )
                  }
                </div>
              </div>
            )
          : null
      }


      {
        !loading &&
        sessions.length ===
          0
          ? (
              <div
                className={
                  styles.empty
                }
              >
                {
                  "Aucun workflow enregistr\u00e9."
                }
              </div>
            )
          : null
      }
    </section>
  );
}
